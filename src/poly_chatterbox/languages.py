"""Languages of Chatterbox Multilingual, with the upstream demo prompts and sample sentences.

The code -> name table mirrors ``chatterbox.mtl_tts.SUPPORTED_LANGUAGES``; it is duplicated here so
the app (and the mock backend) can run without importing torch.
"""

from __future__ import annotations

LANGUAGES: dict[str, str] = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ru": "Russian", "sv": "Swedish", "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

_PROMPTS = "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/"

# Per-language reference clip and sample sentence from the upstream multilingual demo app.
SAMPLES: dict[str, dict[str, str]] = {
    "ar": {"audio": _PROMPTS + "ar_f/ar_prompts2.flac", "gender": "female",
           "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."},
    "da": {"audio": _PROMPTS + "da_m1.flac", "gender": "male",
           "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."},
    "de": {"audio": _PROMPTS + "de_f1.flac", "gender": "female",
           "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."},
    "el": {"audio": _PROMPTS + "el_m.flac", "gender": "male",
           "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."},
    "en": {"audio": _PROMPTS + "en_f1.flac", "gender": "female",
           "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."},
    "es": {"audio": _PROMPTS + "es_f1.flac", "gender": "female",
           "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."},
    "fi": {"audio": _PROMPTS + "fi_m.flac", "gender": "male",
           "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."},
    "fr": {"audio": _PROMPTS + "fr_f1.flac", "gender": "female",
           "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."},
    "he": {"audio": _PROMPTS + "he_m1.flac", "gender": "male",
           "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."},
    "hi": {"audio": _PROMPTS + "hi_f1.flac", "gender": "female",
           "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"},
    "it": {"audio": _PROMPTS + "it_m1.flac", "gender": "male",
           "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."},
    "ja": {"audio": _PROMPTS + "ja/ja_prompts1.flac", "gender": "unknown",
           "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"},
    "ko": {"audio": _PROMPTS + "ko_f.flac", "gender": "female",
           "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."},
    "ms": {"audio": _PROMPTS + "ms_f.flac", "gender": "female",
           "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."},
    "nl": {"audio": _PROMPTS + "nl_m.flac", "gender": "male",
           "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal."},
    "no": {"audio": _PROMPTS + "no_f1.flac", "gender": "female",
           "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."},
    "pl": {"audio": _PROMPTS + "pl_m.flac", "gender": "male",
           "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."},
    "pt": {"audio": _PROMPTS + "pt_m1.flac", "gender": "male",
           "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."},
    "ru": {"audio": _PROMPTS + "ru_m.flac", "gender": "male",
           "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."},
    "sv": {"audio": _PROMPTS + "sv_f.flac", "gender": "female",
           "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."},
    "sw": {"audio": _PROMPTS + "sw_m.flac", "gender": "male",
           "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."},
    "tr": {"audio": _PROMPTS + "tr_m.flac", "gender": "male",
           "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."},
    "zh": {"audio": _PROMPTS + "zh_f2.flac", "gender": "female",
           "text": "上个月，我们达到了一个新的里程碑. 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"},
}

_BY_NAME = {name.lower(): code for code, name in LANGUAGES.items()}


def normalize_language(language: str | None, default: str = "en") -> str:
    """Language code from a code or name, case-insensitive ("French", "fr", "FR" -> "fr")."""
    if not language or not language.strip():
        return default
    key = language.strip().lower()
    code = key if key in LANGUAGES else _BY_NAME.get(key)
    if code is None:
        raise ValueError(f"Unsupported language {language!r}. Supported: {', '.join(LANGUAGES)}.")
    return code
