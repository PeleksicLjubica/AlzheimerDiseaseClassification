"""

File containing constants and functions that are used from different modules.

"""

PATH_TO_TAGGED_FILES = "../TaggedTexts"
PATH_FOR_LOG_FILE = "../TaggedTexts/logs"
PATH_TO_TEXTS = "../Texts" #used for cosine distance counting

REGEX = r'([0-9a-zA-Z{_`’<–”“„\/\(\)\[\]\-\'":*!.?…},šđčćžŠĐÐČĆŽ]*)	([0-9a-zA-Z{_`<–’„”“\/\(\)\[\]\-\'":*!.?…},šđčćžŠÐĐČĆŽ]*)	([0-9a-zA-Z{_`<–’„”“\/\(\)\[\]\-\'":*!.?…},šđčćžŠĐÐČĆŽ]*)'
REGEX_READ_FILE = r'[a-zA-Z\'_ ]*:?: ([0-9.]+)'