import asyncio
from pathlib import Path

from config import Config
from downloader import ChapterDownloader


async def main():
    # ========== КОНФИГУРАЦИЯ ==========
    cfg = Config(
        manga_slug="114307--kaoru-hana-wa-rinto-saku", #https://mangalib.me/ru/manga/ВОТ ЗДЕСЬ БУДУТ ЦИФРЫ И НАЗВАНИЕ, СКОПИРОВАТЬ ДО ВОПРОСИТЕЛЬНОГО ЗНАКА ВКЛЮЧИТЕЛЬНО (если есть)
        chapter_range=(1, 2),  # (начальная глава, конечная глава)
        extra_chapters=[],  # дополнительные главы, например 0.5 или 10.1
        series_title_override="Kaoru Hana Wa Rin To Saku",
        
        # Параметры производительности
        max_concurrent_chapters=1,  # рекомендуется: 1–5
        max_concurrent_images=5,    # рекомендуется: 2–10
        request_delay=0.8,          # рекомендуется: 0.5-5

        # Дополнительные параметры
        output_dir=Path("downloads"),
        cleanup_temp=True,
        volume_override=None,
    )
    # ========== КОНФИГУРАЦИЯ ==========

    downloader = ChapterDownloader(cfg)
    await downloader.download_chapters(cfg.chapter_range)


if __name__ == "__main__":
    asyncio.run(main())
