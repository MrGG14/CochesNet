import asyncio
import pandas as pd
import random
from playwright.async_api import async_playwright


async def main():
    base_url = (
        "https://www.coches.net/search/?MakeIds%5B0%5D=7&ModelIds%5B0%5D=282&Versions%5B0%5D=&pg="
    )
    df = pd.DataFrame(
        columns=["name", "price", "fuel_type", "year", "kms", "cv", "location", "url"]
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.7151.120 Safari/537.36",
            locale="es-ES",
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        # Evasión básica de bots
        await page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', {get: () => ['es-ES', 'es']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        """
        )

        page_num = 1
        while True:
            url = f"{base_url}{page_num}"
            await page.goto(url)
            print(f"Cargando página {page_num}...")

            await page.wait_for_timeout(random.uniform(5000, 10000))

            # Aceptar cookies solo en la primera página
            if page_num == 1:
                try:
                    await page.click("#didomi-notice-agree-button", timeout=3000)
                    print("Se hizo clic en el botón de aceptar cookies")
                except Exception:
                    print("No se encontró el botón de aceptar cookies")

            # Scroll incremental
            last_height = await page.evaluate("document.body.scrollHeight")
            current_position = 0
            scroll_increment = 400
            while current_position < last_height:
                await page.evaluate(f"window.scrollTo(0, {current_position});")
                await page.wait_for_timeout(random.uniform(1000, 1200))
                current_position += scroll_increment
                last_height = await page.evaluate("document.body.scrollHeight")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            await page.wait_for_timeout(random.uniform(2000, 5000))

            # Extraer anuncios
            car_ads = await page.query_selector_all(".mt-CardAd-infoContainer")
            if not car_ads:
                print("No se encontraron más anuncios. Finalizando...")
                break

            for add in car_ads:
                try:
                    name = await add.query_selector_eval(
                        ".mt-CardAd-infoHeaderTitleLink", "el => el.textContent"
                    )
                    price = await add.query_selector_eval(
                        ".mt-CardAdPrice-cashAmount", "el => el.textContent"
                    )
                    attributes = await add.query_selector_all(".mt-CardAd-attrItem")
                    attr_texts = [await attr.text_content() for attr in attributes]
                    fuel_type = attr_texts[0] if len(attr_texts) > 0 else "N/A"
                    year = attr_texts[1] if len(attr_texts) > 1 else "N/A"
                    kms = attr_texts[2] if len(attr_texts) > 2 else "N/A"
                    cv = attr_texts[3] if len(attr_texts) > 3 else "N/A"
                    location = attr_texts[4] if len(attr_texts) > 4 else "N/A"
                    url = await add.query_selector_eval(
                        ".mt-CardAd-infoHeaderTitleLink", "el => el.href"
                    )
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                [
                                    {
                                        "name": name.strip() if name else "",
                                        "price": price.strip() if price else "",
                                        "fuel_type": fuel_type,
                                        "year": year,
                                        "kms": kms,
                                        "cv": cv,
                                        "location": location,
                                        "url": url,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                except Exception as e:
                    print(f"Error al procesar un anuncio: {e}")

            page_num += 1

        await browser.close()
    print(df)
    df.to_csv("coches_net_playwright.csv", index=False, encoding="utf-8")


asyncio.run(main())
