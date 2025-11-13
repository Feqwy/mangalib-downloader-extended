import asyncio
from pathlib import Path
from config import Config
from downloader import ChapterDownloader


def prompt_user_config() -> Config:
    print("=== MangaLib Downloader Config ===\n")

    manga_url = input("Введите ссылку на мангу: ").strip()
    # Извлекаем slug из ссылки, например "https://mangalib.me/114307--kaoru-hana-wa-rinto-saku"
    manga_slug = manga_url.split("/")[-1].split("?")[0]

    start = int(input("Введите начальную главу: ").strip() or "1")
    end = int(input("Введите конечную главу: ").strip() or str(start))

    extra_raw = input("Дополнительные главы (например: 0.5, 10.1, можно оставить пустым): ").strip()
    extra_chapters = [float(x) for x in extra_raw.split(",") if x.strip()] if extra_raw else []

    title_override = input("Название манги (Enter — оставить по умолчанию): ").strip() or None

    # Можно добавить дополнительные параметры
    try:
        max_chapters = int(input("Максимум одновременно загружаемых глав (по умолчанию 1): ") or "1")
        max_images = int(input("Максимум одновременно загружаемых изображений (по умолчанию 5): ") or "5")
        delay = float(input("Задержка между запросами (по умолчанию 0.8): ") or "0.8")
    except ValueError:
        max_chapters, max_images, delay = 1, 5, 0.8

    cfg = Config(
        manga_slug=manga_slug,
        chapter_range=(start, end),
        extra_chapters=extra_chapters,
        series_title_override=title_override,
        max_concurrent_chapters=max_chapters,
        max_concurrent_images=max_images,
        request_delay=delay,
        output_dir=Path("downloads"),
        cleanup_temp=True
    )

    return cfg


async def main():
    cfg = prompt_user_config()
    downloader = ChapterDownloader(cfg)
    await downloader.download_chapters(cfg.chapter_range)


if __name__ == "__main__":
    asyncio.run(main())
