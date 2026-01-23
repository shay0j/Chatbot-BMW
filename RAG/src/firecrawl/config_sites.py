# RAG/src/firecrawl/config_sites.py

SITE_CONFIG = {
    "www.bmw.pl": {
        # Dla BMW.pl raczej crawler statyczny NIE jest skuteczny na wszystkich
        # podstronach konfiguratora — wiele jest renderowanych przez JS.
        # Dlatego domyślne selektory są tutaj zostawione „najprostsze”:
        "model_links": "a[href*='/pl/samochody']",
        "model_name": "h1",
        "description": "meta[name='description']",
        "spec_table": "table tr"
    },

    "bmw.pl": {
        "model_links": "a[href*='/pl/samochody']",
        "model_name": "h1",
        "description": "meta[name='description']",
        "spec_table": "table tr"
    },

    "www.bmw-zkmotors.pl": {
        # na www.bmw-zkmotors.pl lista modeli jest często w linkach rodzaju "Dowiedz się więcej"
        "model_links": "a[href*='/samochody-nowe']",
        "model_name": "h1",
        "description": "meta[name='description']",
        "spec_table": "table tr"
    },

    "bmw-zkmotors.pl": {
        "model_links": "a[href*='/samochody-nowe']",
        "model_name": "h1",
        "description": "meta[name='description']",
        "spec_table": "table tr"
    },

    "www.zkmotors.pl": {
        # zkmotors.pl posiada listę pojazdów (BMW i MINI) w sekcjach,
        # ale raczej działają one przez JS lub framework sklepu,
        # więc selektory poniżej są do ogólnego zestawu linków:
        "model_links": "a[href*='/pojazd']",
        "model_name": "h1",
        "description": "meta[name='description']",
        "spec_table": "table tr"
    },

    "zkmotors.pl": {
        "model_links": "a[href*='/pojazd']",
        "model_name": "h1",
        "description": "meta[name='description']",
        "spec_table": "table tr"
    },

    # MINI – jeśli statycznie znajdziesz jakiekolwiek modele,
    # Playwright obsłuży dynamicznie całą stronę
    "www.mini.com.pl": {
        "model_links": "a[href*='/pl_PL/range']",
        "model_name": ".hero__title, h1",
        "description": "meta[name='description']",
        "spec_table": "table tr"
    },

    "mini.com.pl": {
        "model_links": "a[href*='/pl_PL/range']",
        "model_name": ".hero__title, h1",
        "description": "meta[name='description']",
        "spec_table": "table tr"
    }
}
