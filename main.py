import asyncio
import aiohttp
import json
import time
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from tqdm.asyncio import tqdm as async_tqdm

@dataclass
class Config:
    manga_slug: str
    chapter_range: Tuple[int, int]
    volume_override: Optional[int] = None
    output_dir: Path = Path("downloads")
    max_concurrent_chapters: int = 3
    max_concurrent_images: int = 8
    request_delay: float = 0.03
    fallback_volume_range: Tuple[int, int] = (1, 15)
    cleanup_temp: bool = True
    api_base: str = "https://api.cdnlibs.org/api/manga"
    image_host: str = "https://img3.mixlib.me"
    referer: str = "https://mangalib.me/"

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAG = "\033[95m"

    @staticmethod
    def s(m: str) -> str:
        return f"{C.GREEN}Success: {C.RESET} {m}"

    @staticmethod
    def i(m: str) -> str:
        return f"{C.CYAN}Info: {C.RESET} {m}"

    @staticmethod
    def e(m: str) -> str:
        return f"{C.RED}Error: {C.RESET} {m}"

    @staticmethod
    def chap(n: int) -> str:
        return f"{C.BOLD}{C.MAG}Chapter {n}{C.RESET}"

    @staticmethod
    def title(t: str) -> str:
        return f"{C.BOLD}{C.BLUE}{t}{C.RESET}"

@dataclass
class ChapterInfo:
    number: int
    volume: int
    name: str
    pages_count: int
    series_title: str
    teams: List[str]
    chapter_id: Optional[str] = None

class MangaAPIClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None
        self._chapters_map: Dict[str, Dict[float, int]] = {}
        self._series_cache: Dict[str, Dict[str, Any]] = {}
        self._headers = {
            "User-Agent": "Mozilla/5.0 (iPad; CPU OS 18_6_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/142.0.7444.46 Mobile/15E148 Safari/604.1",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9"
        }

    async def __aenter__(self):
        conn = aiohttp.TCPConnector(limit=self.cfg.max_concurrent_images * 2)
        self._session = aiohttp.ClientSession(connector=conn, headers=self._headers)
        await self._warm()
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    async def _warm(self):
        try:
            async with self._session.get(self.cfg.referer, timeout=6):
                pass
        except Exception:
            pass

    async def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None, retries: int = 5) -> Dict[str, Any]:
        for attempt in range(retries):
            try:
                async with self._session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 429:
                        wait = self._get_retry_after(resp.headers, attempt)
                        await asyncio.sleep(wait)
                        continue

                    resp.raise_for_status()
                    data = await resp.json()
                    await asyncio.sleep(self.cfg.request_delay)
                    return data

            except aiohttp.ClientResponseError as cre:
                if cre.status == 429:
                    await asyncio.sleep(min(2 ** attempt, 60) + 0.1 * attempt)
                    continue
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(0.2 * (attempt + 1))

            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(0.2 * (attempt + 1))

        raise RuntimeError("Retries exhausted")

    @staticmethod
    def _get_retry_after(headers: Dict[str, str], attempt: int) -> float:
        ra = headers.get("Retry-After")
        if ra and ra.isdigit():
            return float(ra)
        return min(2 ** attempt, 60)

    @staticmethod
    def _to_float(s: str) -> Optional[float]:
        try:
            return float(s)
        except ValueError:
            try:
                return float(s.replace(",", "."))
            except ValueError:
                return None

    async def fetch_chapters_list(self, slug: str) -> Dict[float, int]:
        if slug in self._chapters_map:
            return self._chapters_map[slug]

        url = f"{self.cfg.api_base}/{slug}/chapters"
        mapping: Dict[float, int] = {}

        try:
            data = await self._get_json(url, retries=4)
            items = data.get("data", []) if isinstance(data, dict) else []

            for item in items:
                num = item.get("number")
                vol = item.get("volume")
                if num is None or vol is None:
                    continue

                nf = self._to_float(str(num))
                try:
                    vi = int(vol)
                except (ValueError, TypeError):
                    continue

                if nf is not None:
                    mapping[nf] = vi
        except Exception:
            mapping = {}

        self._chapters_map[slug] = mapping
        return mapping

    async def fetch_series_info(self, slug: str) -> Dict[str, Any]:
        if slug in self._series_cache:
            return self._series_cache[slug]

        url = f"{self.cfg.api_base}/{slug}"
        try:
            data = await self._get_json(url, retries=3)
            result = data.get("data", {}) if isinstance(data, dict) else {}
        except Exception:
            result = {}

        self._series_cache[slug] = result
        return result

    async def fetch_chapter_data(self, slug: str, chapter_num: int, volume: int) -> Dict[str, Any]:
        url = f"{self.cfg.api_base}/{slug}/chapter"
        return await self._get_json(
            url,
            params={"number": chapter_num, "volume": volume},
            retries=4
        )

    async def resolve_volume(self, slug: str, chapter_num: int) -> int:
        if self.cfg.volume_override is not None:
            return self.cfg.volume_override

        cmap = await self.fetch_chapters_list(slug)
        target = float(chapter_num)

        if cmap and target in cmap:
            return cmap[target]

        meta = await self.fetch_series_info(slug)

        def search(obj) -> Optional[int]:
            if isinstance(obj, dict):
                num = obj.get("number") or obj.get("chapter_number")
                vol = obj.get("volume")
                if num is not None and vol is not None:
                    nf = self._to_float(str(num))
                    if nf == target:
                        try:
                            return int(vol)
                        except (ValueError, TypeError):
                            pass
                for value in obj.values():
                    result = search(value)
                    if result is not None:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    result = search(item)
                    if result is not None:
                        return result
            return None

        detected = search(meta)
        if detected is not None:
            try:
                await self.fetch_chapter_data(slug, chapter_num, detected)
                return detected
            except Exception:
                pass

        start, end = self.cfg.fallback_volume_range
        for vol in range(start, end + 1):
            try:
                await asyncio.sleep(0.12)
                await self.fetch_chapter_data(slug, chapter_num, vol)
                return vol
            except Exception:
                continue

        raise ValueError(f"Could not determine volume for chapter {chapter_num}")
    
    async def download_image(self, url: str, dest: Path, retries: int = 3):
        headers = {
            **self._headers,
            "Referer": self.cfg.referer,
            "Origin": self.cfg.referer.rstrip("/")
        }

        for attempt in range(retries):
            try:
                async with self._session.get(url, headers=headers, timeout=60) as resp:
                    if resp.status == 403 and attempt < retries - 1:
                        await self._warm()
                        await asyncio.sleep(0.3 * (attempt + 1))
                        continue

                    resp.raise_for_status()
                    data = await resp.read()
                    if not data:
                        raise RuntimeError("Empty response")

                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                    await asyncio.sleep(self.cfg.request_delay)
                    return

            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(0.2 * (attempt + 1))

class ChapterDownloader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize(s: str) -> str:
        s = s.strip()
        s = re.sub(r'[\\/*?:"<>|]', '_', s)
        return s[:200]

    @staticmethod
    def _build_image_url(p: str, host: str) -> str:
        if not p:
            raise ValueError("Empty image path")
        if p.startswith("//"):
            p = p[1:]
        if p.startswith("http"):
            return p
        if not p.startswith("/"):
            p = "/" + p
        return host + p

    async def download_chapter(self, api: MangaAPIClient, ch: int) -> Optional[Path]:
        tmp_dir = self.cfg.output_dir / f"_tmp_ch{ch}_{int(time.time())}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
    
        try:
            volume = await api.resolve_volume(self.cfg.manga_slug, ch)
            chapter_json = await api.fetch_chapter_data(self.cfg.manga_slug, ch, volume)
            data = chapter_json.get("data", {})
    
            if not isinstance(data, dict):
                raise ValueError("Invalid API response: 'data' is not a dictionary")
    
            pages = data.get("pages", [])
            if not pages:
                raise ValueError("No pages found")
    
            meta = await api.fetch_series_info(self.cfg.manga_slug)
            title = meta.get("name") or meta.get("title") or data.get("manga_id") or "Unknown"
            name = str(data.get("name") or "").strip()  # Гарантированная инициализация
            teams = [t.get("name", "") for t in data.get("teams", []) if isinstance(t, dict)]
    
            print(f"\n{C.chap(ch)} | {C.title(title)}")
            print(f"  Volume: {volume} | Pages: {len(pages)} | Name: {name or 'N/A'}")
    
            urls = [
                self._build_image_url(p.get("url") or p.get("image", ""), self.cfg.image_host)
                for p in pages
                if isinstance(p, dict)
            ]
    
            if not urls:
                raise ValueError("No valid image URLs found")
    
            sem = asyncio.Semaphore(self.cfg.max_concurrent_images)
    
            async def download_task(idx: int, url: str):
                async with sem:
                    ext = Path(url).suffix or ".jpg"
                    filename = f"{idx:03d}{ext}"
                    await api.download_image(url, tmp_dir / filename)
    
            tasks = [download_task(i + 1, u) for i, u in enumerate(urls)]
            await async_tqdm.gather(*tasks, desc=f"  Downloading Ch{ch}", unit="img")
    
            info = ChapterInfo(
                number=ch,
                volume=volume,
                name=name,
                pages_count=len(urls),
                series_title=title,
                teams=teams,
                chapter_id=str(data.get("id", ""))
            )
    
            cbz_path = self._create_cbz(tmp_dir, info)
            print(C.s(f"Saved: {cbz_path.name}"))
            return cbz_path
    
        except Exception as e:
            print(C.e(f"Chapter {ch}: {e}"))
            return None
    
        finally:
            if self.cfg.cleanup_temp and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _create_cbz(self, tmp_dir: Path, info: ChapterInfo) -> Path:
        sanitized_name = self._sanitize(
            f"{info.name} - Vol{info.volume} - Ch{info.number}".strip()
        )
        cbz_path = self.cfg.output_dir / f"{sanitized_name}.cbz"

        meta = {
            "series": info.series_title,
            "chapter_number": info.number,
            "volume": info.volume,
            "chapter_name": info.name,
            "chapter_id": info.chapter_id,
            "teams": info.teams,
            "pages": info.pages_count,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with zipfile.ZipFile(cbz_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("info.txt", json.dumps(meta, ensure_ascii=False, indent=2))
            for file in sorted(tmp_dir.iterdir()):
                if file.is_file():
                    zf.write(file, arcname=file.name)

        return cbz_path

    async def download_chapters(self, chapter_range: Tuple[int, int]) -> List[Path]:
        start, end = chapter_range
        chapters = list(range(start, end + 1))

        print(f"\n{C.BOLD}╔══════════════════════════════════════════╗{C.RESET}")
        print(f"{C.BOLD}║      MangaLib Downloader v2.0            ║{C.RESET}")
        print(f"{C.BOLD}╚══════════════════════════════════════════╝{C.RESET}")
        print(f"\n{C.i(f'Manga: {self.cfg.manga_slug}')}")
        print(f"{C.i(f'Chapters: {start}-{end} ({len(chapters)} total)')}")
        print(f"{C.i(f'Concurrency: {self.cfg.max_concurrent_chapters} chapters, {self.cfg.max_concurrent_images} images')}\n")

        async with MangaAPIClient(self.cfg) as api:
            try:
                await api.fetch_chapters_list(self.cfg.manga_slug)
            except Exception:
                pass

            sem = asyncio.Semaphore(self.cfg.max_concurrent_chapters)

            async def download_with_limit(ch: int):
                async with sem:
                    return await self.download_chapter(api, ch)

            results = await asyncio.gather(
                *[download_with_limit(ch) for ch in chapters],
                return_exceptions=True
            )

        successful = [r for r in results if isinstance(r, Path)]
        failed = len(chapters) - len(successful)

        print(f"\n{C.BOLD}{'═' * 50}{C.RESET}")
        print(C.s(f"Completed: {len(successful)}/{len(chapters)} chapters"))
        if failed:
            print(C.i(f"Failed: {failed} chapters"))
        print(C.i(f"Output directory: {self.cfg.output_dir.absolute()}"))
        print(f"{C.BOLD}{'═' * 50}{C.RESET}\n")

        return successful

async def main():
    cfg = Config(
        manga_slug="114307--kaoru-hana-wa-rinto-saku",
        chapter_range=(37, 60),
        max_concurrent_chapters=5,
        max_concurrent_images=10,
        request_delay=0.1
    )

    downloader = ChapterDownloader(cfg)
    await downloader.download_chapters(cfg.chapter_range)


if __name__ == "__main__":
    asyncio.run(main())