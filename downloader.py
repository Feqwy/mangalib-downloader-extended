import asyncio
import json
import time
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
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
    def _format_chapter_number(num) -> str:
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

    def _build_final_chapter_dir(self, series_title: str, volume: int, chapter_num: int) -> Path:
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
    
    async def _get_series_title(self, api: MangaAPIClient, chapter_data: dict) -> str:
        # 1. override
        if self.cfg.series_title_override:
            return self.cfg.series_title_override

        # 2.
        series_block = chapter_data.get("series") or {}
        if isinstance(series_block, dict):
            name = (
                series_block.get("name")
                or series_block.get("rus_name")
                or series_block.get("eng_name")
            )
            if name:
                return name

        # 3.
        try:
            series_meta = await api.fetch_series_info(self.cfg.manga_slug)
            name = (
                series_meta.get("name")
                or series_meta.get("rus_name")
                or series_meta.get("eng_name")
            )
            if name:
                return name
        except Exception:
            pass

        # 4. Fallback
        return self.cfg.manga_slug

    
    async def download_chapter(self, api: MangaAPIClient, chapter_num: int) -> Optional[Tuple[Path, ChapterInfo]]:
        """
        - Если pack_cbz == True -> создаёт tmp_dir и скачивает туда (потом cbz).
        - Если pack_cbz == False -> скачивает сразу в final_chapter_dir.
        Возвращает (dir_used, ChapterInfo) или None при ошибке.
        """
        tmp_dir = None
        try:
            # Определяем том (может делать запросы внутрь)
            volume = await api.resolve_volume(self.cfg.manga_slug, chapter_num)

            chapter_json = await api.fetch_chapter_data(self.cfg.manga_slug, chapter_num, volume)
            if not isinstance(chapter_json, dict):
                raise ValueError("Invalid chapter JSON")

            data = chapter_json.get("data", {})
            if not isinstance(data, dict):
                raise ValueError("Invalid API response: 'data' is not a dictionary")

            pages = data.get("pages", []) or []
            if not pages:
                raise ValueError("No pages found")

            # Узнаём название серии (для построения финальных путей)
            series_title = await self._get_series_title(api, data)
            chapter_name = self.clean_chapter_name(str(data.get("name") or "").strip())

            # где сохраняем: tmp (для CBZ) или финал (для папки)
            if self.cfg.pack_cbz:
                tmp_dir = self.cfg.output_dir / f"_tmp_ch{chapter_num}_{int(time.time())}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                target_dir = tmp_dir
            else:
                target_dir = self._build_final_chapter_dir(series_title, volume, chapter_num)

            # Собираем URL-ы картинок
            urls = [
                self.build_image_url(p.get("url") or p.get("image", ""), self.cfg.image_host)
                for p in pages if isinstance(p, dict)
            ]
            if not urls:
                raise ValueError("No valid image URLs found")

            await self._download_images(api, urls, target_dir, chapter_num)

            teams = [t.get("name", "") for t in data.get("teams", []) if isinstance(t, dict)]

            info = ChapterInfo(
                number=chapter_num,
                volume=volume,
                name=chapter_name,
                pages_count=len(urls),
                series_title=series_title,
                teams=teams,
                chapter_id=str(data.get("id", ""))
            )

            return target_dir, info

        except Exception as e:
            print(Colors.error(f"Chapter {chapter_num}: {e}"))
            # если создали tmp_dir — чистим
            try:
                if tmp_dir and tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            return None


    async def _download_images(self, api: MangaAPIClient, urls: List[str], chapter_dir: Path, chapter_num: int):
        """
        Параллельная загрузка страниц в указанный каталог.
        Не создаёт tmp для pack_cbz=False, потому chapter_dir должен существовать.
        """
        sem = asyncio.Semaphore(self.cfg.max_concurrent_images)

        async def download_task(idx: int, url: str):
            async with sem:
                try:
                    image_bytes, content_type = await api.download_image_raw(url)
                except Exception as e:
                    print(Colors.error(f"Failed to download page {idx} for Ch{chapter_num}: {e}"))
                    return

                ext = Path(url).suffix.lower() or ".jpg"

                if "image/gif" in content_type:
                    try:
                        image_bytes = convert_gif_to_png(image_bytes)
                        ext = ".png"
                    except Exception as e:
                        print(Colors.warning(f"GIF conversion failed for Ch{chapter_num} page {idx}: {e}"))

                dest = chapter_dir / f"{idx:03d}{ext}"
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(image_bytes)
                except Exception as e:
                    print(Colors.error(f"Failed to save page {idx} for Ch{chapter_num}: {e}"))

        tasks = [download_task(i + 1, url) for i, url in enumerate(urls)]
        # tqdm wrapper для параллельных задач
        await async_tqdm.gather(*tasks, desc=f"  Downloading Ch{chapter_num}", unit="img")
    def create_cbz(self, src_dir: Path, info: ChapterInfo, cbz_path: Path):
        """Создать CBZ из содержимого src_dir (файлы в сортированном порядке)."""
        try:
            cbz_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(cbz_path, 'w', compression=zipfile.ZIP_STORED) as zf:
                for entry in sorted(src_dir.iterdir()):
                    if entry.is_file():
                        # Добавляем в архив без абсолютных путей — только имя файла
                        zf.write(entry, arcname=entry.name)
            print(Colors.info(f"Created CBZ: {cbz_path.name}"))
        except Exception as e:
            print(Colors.error(f"Failed to create CBZ {cbz_path}: {e}"))
            raise

    def _process_volumes(self, volume_groups: Dict[int, List[Tuple[Path, ChapterInfo]]],
                         series_folder: Path, series_title: str, series_meta: dict):
        """
        Обработка томов:
        - если pack_cbz=True: из tmp_dir создаются .cbz внутри volume_folder
        - если pack_cbz=False: главы уже на месте, ничего не перемещаем
        """
        for volume in sorted(volume_groups):
            chapter_list = sorted(volume_groups[volume], key=lambda x: ChapterDownloader._format_chapter_number(x[1].number))
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
                chap_str = ChapterDownloader._format_chapter_number(info.number)
                chap_name = f"Chapter {chap_str}"
                sanitized_chap = self.sanitize_filename(chap_name)

                if self.cfg.pack_cbz:
                    # dir_path — tmp_dir: создаём cbz в vol_folder
                    cbz_path = vol_folder / f"{sanitized_chap}.cbz"
                    try:
                        self.create_cbz(dir_path, info, cbz_path)
                    except Exception as e:
                        print(Colors.error(f"Failed to create CBZ for chapter {info.number}: {e}"))
                    # удаляем tmp
                    try:
                        if dir_path.exists():
                            shutil.rmtree(dir_path, ignore_errors=True)
                    except Exception:
                        pass
                else:
                    # pack_cbz == False -> dir_path уже указывает на финальную папку внутри downloads/<Series>/Volume XX/Chapter YY
                    # убедимся, что папка существует и при необходимости переместим/переименуем (обычно не нужно)
                    target_folder = vol_folder / sanitized_chap
                    try:
                        if dir_path.exists() and dir_path != target_folder:
                            # если chapter загружен по другой структуре — переместим (редкий случай)
                            if target_folder.exists():
                                shutil.rmtree(target_folder, ignore_errors=True)
                            shutil.move(str(dir_path), str(target_folder))
                        else:
                            # если всё в порядке — убедимся, что структура корректна
                            target_folder.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        print(Colors.warning(f"Could not normalize chapter folder for {info.number}: {e}"))

    def _create_final_archive(self, temp_series_dir: Path, sanitized_series: str) -> Path:
        """
        Создаёт zip архива серии: downloads/<sanitized_series>.zip содержащий внутри
        папку <sanitized_series>/...
        """
        zip_base = self.cfg.output_dir / sanitized_series
        # shutil.make_archive использует path без расширения и создаёт .zip рядом
        # root_dir = temp_series_dir, base_dir = sanitized_series — чтобы внутри zip была именно папка sanitized_series
        shutil.make_archive(str(zip_base), 'zip', root_dir=str(temp_series_dir), base_dir=sanitized_series)
        zip_path = zip_base.with_suffix('.zip')
        print(Colors.success(f"Saved archive: {zip_path.name}"))
        return zip_path

    def _cleanup(self, successful: list, temp_series_dir: Path):
        if not self.cfg.cleanup_temp:
            return
        for tmp_dir, _ in successful:
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        try:
            if temp_series_dir.exists():
                shutil.rmtree(temp_series_dir, ignore_errors=True)
        except Exception:
            pass
    async def download_chapters(self, chapter_range: Tuple[int, int]):
        start, end = chapter_range
        chapters = list(range(start, end + 1))

        async with MangaAPIClient(self.cfg) as api:
            self._print_header(start, end, len(chapters))
            results = await self._download_all_chapters(api, chapters)

            successful, failed_count = self._process_results(chapters, results)
            if not successful:
                print(Colors.error("No chapters downloaded."))
                return

            # Получаем метаданные серии один раз
            series_meta = await api.fetch_series_info(self.cfg.manga_slug)
            series_title = self._determine_series_title(series_meta)
            sanitized_series = self.sanitize_filename(series_title)

            # создаём итоговую структуру
            if not self.cfg.pack_cbz:
                # главы уже скачаны в final folders -> собираем metadata и cover
                series_folder = self.cfg.output_dir / sanitized_series
                series_folder.mkdir(parents=True, exist_ok=True)
                await self._download_series_cover(series_meta, series_folder, api)
                self._create_series_metadata(series_folder, series_title, series_meta)
                # Для безопасности: сгруппируем по томам и вызовем _process_volumes (он не будет перемещать, но обеспечит ComicInfo)
                volume_groups = defaultdict(list)
                for path, info in successful:
                    volume_groups[info.volume].append((path, info))
                self._process_volumes(volume_groups, series_folder, series_title, series_meta)

                print(Colors.success(f"Saved folder: {series_folder}"))
                self._print_summary(len(successful), len(chapters), failed_count)
                return series_folder

            # pack_cbz == True -> создаём временную структуру, разместим cbz в temp_series_dir/sanitized_series/...
            temp_series_dir = self.cfg.output_dir / f"_tmp_series_{int(time.time())}"
            temp_series_dir.mkdir(parents=True, exist_ok=True)
            series_folder = temp_series_dir / sanitized_series
            series_folder.mkdir(parents=True, exist_ok=True)

            await self._download_series_cover(series_meta, series_folder, api)
            self._create_series_metadata(series_folder, series_title, series_meta)

            # группируем по томам и создаём .cbz файлы
            volume_groups = defaultdict(list)
            for tmp_dir, info in successful:
                volume_groups[info.volume].append((tmp_dir, info))

            self._process_volumes(volume_groups, series_folder, series_title, series_meta)

            # упаковываем итоговую папку sanitized_series в downloads/<sanitized_series>.zip
            zip_path = self._create_final_archive(temp_series_dir, sanitized_series)

            # удаляем временные tmp (tmp_dir уже удалены в _process_volumes, но удалим temp_series_dir)
            self._cleanup(successful, temp_series_dir)

            self._print_summary(len(successful), len(chapters), failed_count)
            return zip_path

    async def _download_all_chapters(self, api: MangaAPIClient, chapters: List[int]) -> list:
        sem = asyncio.Semaphore(self.cfg.max_concurrent_chapters)

        async def limited(ch):
            async with sem:
                return await self.download_chapter(api, ch)

        return await asyncio.gather(*[limited(ch) for ch in chapters], return_exceptions=True)

    def _process_results(self, chapters: List[int], results: list) -> Tuple[list, int]:
        successful = []
        failed_count = 0
        for ch, result in zip(chapters, results):
            if isinstance(result, Exception):
                print(Colors.error(f"Chapter {ch}: {result}"))
                failed_count += 1
            elif result:
                successful.append(result)
        return successful, failed_count

    def _determine_series_title(self, series_meta: dict) -> str:
        return (
            self.cfg.series_title_override
            or series_meta.get("name")
            or series_meta.get("rus_name")
            or series_meta.get("eng_name")
            or self.cfg.manga_slug
        )

    async def _download_series_cover(self, series_meta: dict, series_folder: Path, api: MangaAPIClient):
        cover = series_meta.get("cover", {}) or {}
        cover_url = cover.get("default") if isinstance(cover, dict) else cover
        if not cover_url:
            return
        try:
            data, _ct = await api.download_image_raw(cover_url)
            for name in ("Series Cover.jpg", "cover.jpg"):
                (series_folder / name).write_bytes(data)
                break
        except Exception:
            pass

    def _create_series_metadata(self, series_folder: Path, series_title: str, series_meta: dict):
        if not self.cfg.generate_metadata:
            return
        try:
            series_json = self.metadata_gen.create_series_json(series_title, series_meta)
            (series_folder / "series.json").write_text(series_json, encoding="utf-8")
        except Exception:
            pass

    def _print_header(self, start: int, end: int, total: int):
        print(f"\n{Colors.BOLD}╔══════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}║         MangaLib Downloader v2.0         ║{Colors.RESET}")
        print(f"{Colors.BOLD}╚══════════════════════════════════════════╝{Colors.RESET}")
        series_info = self.cfg.series_title_override or f"{self.cfg.manga_slug} (from slug)"
        print(f"\n{Colors.info(f'Manga: {series_info}')}")
        print(f"{Colors.info(f'Chapters: {start}-{end} ({total} total)')}")
        print(f"{Colors.info(f'Concurrency: {self.cfg.max_concurrent_chapters} chapters, {self.cfg.max_concurrent_images} images')}\n")

    def _print_summary(self, successful: int, total: int, failed: int):
        print(f"\n{Colors.BOLD}{'═' * 50}{Colors.RESET}")
        print(Colors.success(f"Completed: {successful}/{total} chapters"))
        if failed:
            print(Colors.info(f"Failed: {failed} chapters"))
        print(Colors.info(f"Output directory: {self.cfg.output_dir.absolute()}"))
        print(f"{Colors.BOLD}{'═' * 50}{Colors.RESET}\n")