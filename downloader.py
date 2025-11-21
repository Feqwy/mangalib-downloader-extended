import asyncio
import json
import time
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from collections import defaultdict
from tqdm.asyncio import tqdm as async_tqdm

from config import Config
from colors import Colors
from models import ChapterInfo
from api_client import MangaAPIClient
from metadata import MetadataGenerator

from PIL import Image
import io


def convert_gif_to_png(image_bytes: bytes) -> bytes:
    """Convert GIF → PNG (fixes problematic GIFs)."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("P", "RGBA"):
        img = img.convert("RGB")
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


class ChapterDownloader:
    @staticmethod
    def _format_chapter_number(num: Union[int, float, str]) -> str:
        if isinstance(num, str):
            return num
        try:
            f = float(num)
            if f.is_integer():
                return str(int(f))
            s = str(f).rstrip("0").rstrip(".")
            return s
        except Exception:
            return str(num)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.metadata_gen = MetadataGenerator(cfg)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_filename(text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r'[\\/*?:"<>|]', "_", text)
        return text[:200]

    def _build_final_chapter_dir(self, series_title: str, volume: int, chapter_num: Union[int, float]) -> Path:
        sanitized_series = self.sanitize_filename(series_title)
        vol_name = f"Volume {int(volume):02d}"
        chap_str = ChapterDownloader._format_chapter_number(chapter_num)
        chapter_name = f"Chapter {chap_str}"

        final_dir = self.cfg.output_dir / sanitized_series / vol_name / chapter_name
        final_dir.mkdir(parents=True, exist_ok=True)
        return final_dir

    @staticmethod
    def build_image_url(path: str, host: str) -> str:
        if not path:
            raise ValueError("Empty image path")
        if path.startswith("//"):
            path = path[1:]
        if path.startswith("http"):
            return path
        if not path.startswith("/"):
            path = "/" + path
        return host + path

    @staticmethod
    def clean_chapter_name(name: str) -> str:
        if not name:
            return ""
        name = re.sub(r'\s*\([^)]*\d[^)]*\)', '', name).strip()
        name = re.sub(r'\d+', '', name).strip()
        return name

    # Логика получения списка глав (индексный подход)
    async def _resolve_chapter_list(self, api: MangaAPIClient) -> List[ChapterInfo]:
        start, end = self.cfg.chapter_range
        chapters = await api.to_chapter_info_list(
            slug=self.cfg.manga_slug,
            start_num=float(start),
            end_num=float(end),
            extra=self.cfg.extra_chapters or []
        )
        print(Colors.info(f"Найдено глав для загрузки (после проверки API): {len(chapters)}"))
        return chapters

    async def download_chapters(self):
        start, end = self.cfg.chapter_range
        self._print_header(start, end)

        async with MangaAPIClient(self.cfg) as api:
            try:
                chapters_to_download = await self._resolve_chapter_list(api)
            except Exception as e:
                print(Colors.error(f"Ошибка при определении списка глав: {e}"))
                return

            if not chapters_to_download:
                print(Colors.warning("Список глав для загрузки пуст. Проверьте диапазон."))
                return

            series_meta = await api.fetch_series_info(self.cfg.manga_slug)

            results = await self._download_all_chapters(api, chapters_to_download)

            successful, failed_count = self._process_results(chapters_to_download, results)

            if not successful:
                print(Colors.error("No chapters downloaded successfully"))
                return []

            series_title = self._determine_series_title(series_meta)
            zip_path = await self._create_series_archive(successful, series_title, series_meta)

            self._print_summary(len(successful), len(chapters_to_download), failed_count)
            return [zip_path] if zip_path else []

    async def download_chapter(self, api: MangaAPIClient, chapter_item: Union[ChapterInfo, int, float]) -> Optional[Tuple[Path, ChapterInfo]]:
        """
        Скачивает главу. Принимает ChapterInfo (предпочтительно) или число.
        """
        tmp_dir = None

        # Извлекаем данные из переданного объекта
        if isinstance(chapter_item, ChapterInfo):
            chapter_num = chapter_item.number
            req_number = getattr(chapter_item, "number_str", chapter_num)
            volume = chapter_item.volume
            chapter_index = chapter_item.index  # Берем индекс из объекта
        else:
            chapter_num = chapter_item
            req_number = chapter_item  # Fallback
            volume = None
            chapter_index = 0  # Fallback, если передано просто число

        try:
            # Если том неизвестен
            if volume is None:
                volume = await api.resolve_volume(self.cfg.manga_slug, chapter_num)

            # Получаем данные страниц
            chapter_json = await api.fetch_chapter_data(self.cfg.manga_slug, req_number, volume)
            if not isinstance(chapter_json, dict):
                raise ValueError("Invalid chapter JSON")

            data = chapter_json.get("data", {})
            if not isinstance(data, dict):
                raise ValueError("Invalid API response")

            pages = data.get("pages", []) or []
            if not pages:
                raise ValueError("No pages found")

            series_title = await self._get_series_title(api, data)
            chapter_name = self.clean_chapter_name(str(data.get("name") or "").strip())

            # Пути сохранения
            if self.cfg.pack_cbz:
                # Используем req_number в имени временной папки
                tmp_dir = self.cfg.output_dir / f"_tmp_ch{req_number}_{int(time.time())}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                target_dir = tmp_dir
            else:
                target_dir = self._build_final_chapter_dir(series_title, volume, req_number)

            urls = [
                self.build_image_url(p.get("url") or p.get("image", ""), self.cfg.image_host)
                for p in pages if isinstance(p, dict)
            ]
            if not urls:
                raise ValueError("No valid image URLs found")

            download_success = await self._download_images(api, urls, target_dir, req_number)

            if not download_success:
                # Если хотя бы одно изображение не скачалось (например, из-за 429),
                # вызываем ошибку, чтобы перейти в блок 'except' и вернуть None.
                raise RuntimeError("Failed to download all required images.")

            teams = [t.get("name", "") for t in data.get("teams", []) if isinstance(t, dict)]

            # Создание ChapterInfo с передачей index
            info = ChapterInfo(
                number=chapter_num,
                number_str=str(req_number), # Сохраняем строку
                index=chapter_index, # Передаем сохраненный индекс
                volume=volume,
                name=chapter_name,
                pages_count=len(urls),
                series_title=series_title,
                teams=teams,
                chapter_id=str(data.get("id", ""))
            )

            return target_dir, info

        except Exception as e:
            print(Colors.error(f"Chapter {req_number}: {e}"))
            try:
                if tmp_dir and tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            return None

    async def _download_images(self, api: MangaAPIClient, urls: List[str], chapter_dir: Path, chapter_num: Union[int, float, str]) -> bool:
        sem = asyncio.Semaphore(self.cfg.max_concurrent_images)

        async def download_task(idx: int, url: str) -> bool:
            async with sem:
                ext = Path(url).suffix.lower() or ".jpg"
                dest = chapter_dir / f"{idx:03d}{ext}"
                try:
                    # api.download_image поднимет исключение, если 429 не удалось обойти
                    await api.download_image(url, dest)
                except Exception as e:
                    print(Colors.error(f"Failed to download page {idx} for Ch{chapter_num}: {e}"))
                    return False

                # Сохранение прошло внутри api.download_image
                return True

        tasks = [download_task(i + 1, url) for i, url in enumerate(urls)]
        # tqdm wrapper для параллельных задач
        results = await async_tqdm.gather(*tasks, desc=f"  Downloading Ch{chapter_num}", unit="img")

        # Проверка результатов на наличие хотя бы одного False
        failed_downloads = sum(1 for res in results if res is not True)

        if failed_downloads > 0:
            print(Colors.error(f"Chapter {chapter_num} failed to download {failed_downloads}/{len(urls)} images."))
            return False

        return True

    async def _download_all_chapters(self, api: MangaAPIClient, chapters: List[Union[ChapterInfo, int]]) -> list:
        sem = asyncio.Semaphore(self.cfg.max_concurrent_chapters)

        async def limited(ch):
            async with sem:
                return await self.download_chapter(api, ch)

        return await asyncio.gather(*[limited(ch) for ch in chapters], return_exceptions=True)

    def create_cbz(self, src_dir: Path, info: ChapterInfo, cbz_path: Path):
        try:
            cbz_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(cbz_path, 'w', compression=zipfile.ZIP_STORED) as zf:
                for entry in sorted(src_dir.iterdir()):
                    if entry.is_file():
                        # Добавляем в архив без абсолютных путей — только имя файла
                        zf.write(entry, arcname=entry.name)
                # print(Colors.info(f"Created CBZ: {cbz_path.name}")) # Можно раскомментировать для отладки
        except Exception as e:
            print(Colors.error(f"Failed to create CBZ {cbz_path}: {e}"))
            raise

    def _process_volumes(self, volume_groups: Dict[int, List[Tuple[Path, ChapterInfo]]],
                         series_folder: Path, series_title: str, series_meta: dict):
        for volume in sorted(volume_groups):
            chapter_list = sorted(volume_groups[volume], key=lambda x: x[1].number)
            vol_name = f"Volume {int(volume):02d}"
            vol_folder = series_folder / vol_name
            vol_folder.mkdir(parents=True, exist_ok=True)
            # опционально создаём ComicInfo.xml на уровне тома
            if self.cfg.generate_metadata:
                try:
                    vol_xml = self.metadata_gen.create_volume_comicinfo(volume, series_title, len(chapter_list), series_meta)
                    (vol_folder / "ComicInfo.xml").write_bytes(vol_xml)
                except Exception:
                    pass
            for dir_path, info in chapter_list:
                chap_str = info.number_str if hasattr(info, "number_str") else ChapterDownloader._format_chapter_number(info.number)
                sanitized_chap = self.sanitize_filename(f"Chapter {chap_str}")
                if self.cfg.pack_cbz:
                    # dir_path — tmp_dir: создаём cbz в vol_folder
                    cbz_path = vol_folder / f"{sanitized_chap}.cbz"
                    self.create_cbz(dir_path, info, cbz_path)
                    try:
                        if dir_path.exists():
                            shutil.rmtree(dir_path, ignore_errors=True)
                    except:
                        pass
                else:
                    target_folder = vol_folder / sanitized_chap
                    if dir_path.exists() and dir_path != target_folder:
                        if target_folder.exists():
                            shutil.rmtree(target_folder)
                        shutil.move(str(dir_path), str(target_folder))

    async def _get_series_title(self, api: MangaAPIClient, chapter_data: dict) -> str:
        if self.cfg.series_title_override:
            return self.cfg.series_title_override
        series_block = chapter_data.get("series") or {}
        if isinstance(series_block, dict):
            name = series_block.get("name") or series_block.get("rus_name") or series_block.get("eng_name")
            if name:
                return name
        try:
            series_meta = await api.fetch_series_info(self.cfg.manga_slug)
            return (series_meta.get("name") or series_meta.get("rus_name") or
                    series_meta.get("eng_name") or self.cfg.manga_slug)
        except:
            return self.cfg.manga_slug

    def _process_results(self, chapters: List[Any], results: list) -> Tuple[list, int]:
        successful = []
        failed_count = 0
        for ch, result in zip(chapters, results):
            if isinstance(result, Exception):
                num = ch.number_str if isinstance(ch, ChapterInfo) and hasattr(ch, "number_str") else (ch.number if isinstance(ch, ChapterInfo) else ch)
                print(Colors.error(f"Chapter {num}: {result}"))
                failed_count += 1
            elif result:
                successful.append(result)
        return successful, failed_count

    def _determine_series_title(self, series_meta: dict) -> str:
        return (self.cfg.series_title_override or series_meta.get("name") or
                series_meta.get("rus_name") or series_meta.get("eng_name") or self.cfg.manga_slug)

    async def _download_series_cover(self, series_meta: dict, series_folder: Path, api: MangaAPIClient):
        cover = series_meta.get("cover", {}) or {}
        cover_url = cover.get("default") if isinstance(cover, dict) else cover
        if cover_url:
            try:
                data, _ = await api.download_image_raw(cover_url)
                (series_folder / "cover.jpg").write_bytes(data)
            except Exception:
                try:
                    await api.download_image(cover_url, series_folder / "cover.jpg")
                except Exception:
                    pass

    def _create_series_metadata(self, series_folder: Path, series_title: str, series_meta: dict):
        if self.cfg.generate_metadata:
            try:
                series_json = self.metadata_gen.create_series_json(series_title, series_meta)
                (series_folder / "series.json").write_text(series_json, encoding="utf-8")
                series_xml = self.metadata_gen.create_series_comicinfo(series_title, series_meta)
                (series_folder / "ComicInfo.xml").write_bytes(series_xml)
            except:
                pass

    async def _create_series_archive(self, successful: list, series_title: str, series_meta: dict) -> Optional[Path]:
        sanitized_series = self.sanitize_filename(series_title)
        if not self.cfg.pack_cbz:
            # Если не пакуем в CBZ, просто организуем папки
            series_folder = self.cfg.output_dir / sanitized_series
            series_folder.mkdir(parents=True, exist_ok=True)
            await self._download_series_cover(series_meta, series_folder, MangaAPIClient(self.cfg))
            self._create_series_metadata(series_folder, series_title, series_meta)
            volume_groups = defaultdict(list)
            for path, info in successful:
                volume_groups[info.volume].append((path, info))
            self._process_volumes(volume_groups, series_folder, series_title, series_meta)
            return series_folder

        # Логика для ZIP архива
        temp_series_dir = self.cfg.output_dir / f"_tmp_series_{int(time.time())}"
        temp_series_dir.mkdir(parents=True, exist_ok=True)
        series_folder = temp_series_dir / sanitized_series
        series_folder.mkdir(parents=True, exist_ok=True)
        self._create_series_metadata(series_folder, series_title, series_meta)
        volume_groups = defaultdict(list)
        for tmp_dir, info in successful:
            volume_groups[info.volume].append((tmp_dir, info))
        self._process_volumes(volume_groups, series_folder, series_title, series_meta)
        zip_path = self._create_final_archive(temp_series_dir, sanitized_series)
        self._cleanup(successful, temp_series_dir)
        return zip_path

    def _create_final_archive(self, temp_series_dir: Path, sanitized_series: str) -> Path:
        zip_base = self.cfg.output_dir / sanitized_series
        shutil.make_archive(str(zip_base), 'zip', root_dir=str(temp_series_dir), base_dir=sanitized_series)
        return zip_base.with_suffix('.zip')

    def _cleanup(self, successful: list, temp_series_dir: Path):
        if self.cfg.cleanup_temp:
            if temp_series_dir.exists():
                shutil.rmtree(temp_series_dir, ignore_errors=True)

    def _print_header(self, start: int, end: int):
        print(f"\n{Colors.BOLD}╔══════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}║         MangaLib Downloader v2.0         ║{Colors.RESET}")
        print(f"{Colors.BOLD}╚══════════════════════════════════════════╝{Colors.RESET}")
        series_info = self.cfg.series_title_override or f"{self.cfg.manga_slug} (from slug)"
        print(f"\n{Colors.info(f'Manga: {series_info}')}")
        print(f"{Colors.info(f'Requested Chapters: {start}-{end}')}")
        if self.cfg.extra_chapters:
            print(f"{Colors.info(f'Extra Chapters: {self.cfg.extra_chapters}')}")
        print(f"{Colors.info(f'Concurrency: {self.cfg.max_concurrent_chapters} chapters, {self.cfg.max_concurrent_images} images')}\n")

    def _print_summary(self, successful: int, total: int, failed: int):
        print(f"\n{Colors.BOLD}{'═' * 50}{Colors.RESET}")
        print(Colors.success(f"Completed: {successful}/{total} chapters"))
        if failed:
            print(Colors.info(f"Failed: {failed} chapters"))
        print(Colors.info(f"Output directory: {self.cfg.output_dir.absolute()}"))
        print(f"{Colors.BOLD}{'═' * 50}{Colors.RESET}\n")
