import asyncio
from pathlib import Path
from config import Config
from downloader import ChapterDownloader


def prompt_user_config() -> Config:
    print("=== MangaLib Downloader Config ===\n")

    manga_url = input("Введите ссылку на мангу: ").strip()
    # Извлекаем slug из ссылки
    # Пример: https://mangalib.me/title-slug?section=... -> title-slug
    try:
        manga_slug = manga_url.split("/")[-1].split("?")[0]
        if not manga_slug: # Если ссылка оканчивается на слэш
            manga_slug = manga_url.split("/")[-2]
    except IndexError:
        print("Некорректная ссылка, используется slug по умолчанию")
        manga_slug = "unknown"

    # --- ЛОГИКА ОПРЕДЕЛЕНИЯ САЙТА ---
    is_slash = "v2.shlib.life" in manga_url
    
    if is_slash:
        print("\n[!] Обнаружена ссылка на SlashLib.")
        default_api = "https://hapi.hentaicdn.org/api/manga"
        default_image_host = "https://img3.mixlib.me"
        default_referer = "https://v2.shlib.life/"
        
        # Запрос токена
        print("Для этого API может потребоваться Bearer Token (из Cookies или LocalStorage).")
        auth_token = input("Введите Bearer Token (или Enter, чтобы попробовать без него): ").strip() or None
    else:
        default_api = "https://api.cdnlibs.org/api/manga"
        default_image_host = "https://img3.mixlib.me"
        default_referer = "https://mangalib.me/"
        auth_token = None

    # -------------------------------

    print(f"\nAPI URL: {default_api}")
    print(f"Slug: {manga_slug}\n")

    start = int(input("Введите начальную главу: ").strip() or "1")
    end = int(input("Введите конечную главу: ").strip() or str(start))

    extra_raw = input("Дополнительные главы (например: 0.5, 10.1, можно оставить пустым): ").strip()
    extra_chapters = [float(x) for x in extra_raw.split(",") if x.strip()] if extra_raw else []

    title_override = input("Название манги (Enter — оставить по умолчанию): ").strip() or None

    # Настройки загрузки
    try:
        max_chapters = int(input("Максимум одновременно загружаемых глав (по умолчанию 1): ") or "1")
        max_images = int(input("Максимум одновременно загружаемых изображений (по умолчанию 5): ") or "5")
        delay = float(input("Задержка между запросами (по умолчанию 0.5): ") or "0.5")
    except ValueError:
        max_chapters, max_images, delay = 1, 5, 0.5

    pack_cbz_input = input("Собирать CBZ архивы? (y/n, по умолчанию y): ").strip().lower()
    pack_cbz = pack_cbz_input != "n"

    generate_metadata_input = input("Создавать ComicInfo/series.json? (y/n, по умолчанию y): ").strip().lower()
    generate_metadata = generate_metadata_input != "n"

    cfg = Config(
        manga_slug=manga_slug,
        chapter_range=(start, end),
        extra_chapters=extra_chapters,
        series_title_override=title_override,
        max_concurrent_chapters=max_chapters,
        max_concurrent_images=max_images,
        request_delay=delay,
        output_dir=Path("downloads"),
        cleanup_temp=True,
        pack_cbz=pack_cbz,
        generate_metadata=generate_metadata,
        # Динамические параметры API
        api_base=default_api,
        image_host=default_image_host,
        referer=default_referer,
        auth_token=auth_token
    )

    return cfg


async def main():
    cfg = prompt_user_config()
    downloader = ChapterDownloader(cfg)
    await downloader.download_chapters(cfg.chapter_range)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nЗагрузка прервана пользователем.")