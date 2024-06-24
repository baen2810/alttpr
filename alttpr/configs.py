
# Define the predefined RGB values for each color label
DEFAULT_COLOR_LABELS_MAP_TRACKERS = {
    'DEFAULT': {
        'RED': (230, 0, 0),
        'GREEN': (20, 255, 20),
        'LIGHTBLUE': (40, 180, 240),
        'DARKBLUE': (0, 0, 240),
        'ORANGE': (240, 160, 20),
        'YELLOW': (245, 255, 15),
        'GREY': (128, 128, 128),
        'PURPLE': (128, 0, 128),
    },
}

# Define the predefined RGB values for each item tracker label
DEFAULT_COLOR_LABELS_ITEM_TRACKER = {
    '11|SWO': {
        'NONE': (0, 0, 0),
        'FIGHTER': (248, 248, 248),
        'MASTER': (160, 248, 216),
        'TEMPERED': (248, 160, 40),
        'BUTTER': (248, 248, 200),
    },
    '12|BOW': {
        'NONE': (0, 0, 0),
        'SIMPLE': (248, 176, 80),
        'SILVERS': (248, 248, 248),
    },
    '13|BMR': {
        'NONE': (36, 42, 58),
        'BLUE': (144, 168, 232),
        'RED': (224, 112, 112),
    },
    '14|HKS': {
        'NONE': (44, 10, 10),
        'AVLBL': (176, 40, 40),
    },
    '15|BMB': {
        'NONE': (64, 64, 64),
        'AVLBL': (255, 255, 255),
    },
    '16|MSR': {
        'NONE': (46, 34, 8),
        'AVLBL': (200, 48, 24),
    },
    '17|POW': {
        'NONE': (46, 24, 10),
        'AVLBL': (248, 176, 80),
    },
    '21|MNP': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '22|FIR': {
        'NONE': (56, 28, 28),
        'AVLBL': (224, 112, 112),
    },
    '23|ICR': {
        'NONE': (36, 42, 58),
        'AVLBL': (144, 168, 232),
    },
    '24|BBS': {
        'NONE': (46, 35, 10),
        'AVLBL': (184, 140, 40),
    },
    '25|ETH': {
        'NONE': (46, 35, 10),
        'AVLBL': (184, 140, 40),
    },
    '26|QUK': {
        'NONE': (46, 35, 10),
        'AVLBL': (184, 140, 40),
    },
    '31|EP': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '32|LMP': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '33|HMR': {
        'NONE': (50, 22, 12),
        'AVLBL': (200, 88, 48),
    },
    '34|SVL': {
        'NONE': (46, 46, 50),
        'AVLBL': (148, 148, 200),
    },
    '35|FLU': {
        'NONE': (20, 26, 42),
        'AVLBL': (144, 168, 232),
    },
    '36|BGN': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '37|BOK': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '38|SLD': {
        'NONE': (0, 0, 0),
        'WOOD': (248, 248, 248),
        'IRON': (176, 40, 40),
        'MIRROR': (200, 224, 224),
    },
    '41|DP': {
        'NONE': (16, 32, 22),
        'AVLBL': (80, 192, 144),
    },
    '42|BTL': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '43|SOM': {
        'NONE': (56, 28, 28),
        'AVLBL': (224, 112, 112),
    },
    '44|BYR': {
        'NONE': (36, 42, 58),
        'AVLBL': (144, 168, 232),
    },
    '45|CAP': {
        'NONE': (56, 28, 28),
        'AVLBL': (224, 112, 112),
    },
    '46|MIR': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '51|TH': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '52|BTS': {
        'NONE': (56, 28, 28),
        'AVLBL': (224, 112, 112),
    },
    '53|GLV': {
        'NONE': (62, 62, 62),
        'MITTS': (120, 120, 136),
        'TITANS': (184, 139, 38),
    },
    '54|FLP': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '55|MAG': {
        'NONE': (38, 52, 28),
        'AVLBL': (153, 209, 113),
    },
    '56|AG1': {
        'NONE': (30, 30, 30),
        'AVLBL': (0, 0, 0),
    },
    '57|AG2': {
        'NONE': (30, 30, 30),
        'AVLBL': (0, 0, 0),
    },
    '61|POD': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '62|SP': {
        'NONE': (58, 32, 26),
        'AVLBL': (232, 128, 104),
    },
    '63|SW': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '64|TT': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '65|IP': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '66|MM': {
        'NONE': (62, 62, 62),
        'AVLBL': (248, 248, 248),
    },
    '67|TR': {
        'NONE': (28, 28, 26),
        'AVLBL': (248, 248, 248),
    },
    'DEFAULT': {
        'OFF': (62, 62, 62),
        'ON': (248, 248, 248),
    }
}


# default coordinates for the itemtracker box
DEFAULT_ITEMTRACKER_BOX = (1418, 0, 245, 224)

# default coordinates for the lightworld map tracker box
DEFAULT_LIGHTWORLD_MAP_BOX = (1413, 229, 251, 261)

# default coordinates for the darkworld map tracker box
DEFAULT_DARKWORLD_MAP_BOX = (1670, 229, 249, 260)

# default tracking point coordinates for the itemtracker box
# absolute pixels relative to top left corner of box
DEFAULT_ITEMTRACKER_POINTS = {
    # itemtracker row 1
    '11|SWO': (52, 6),
    '12|BOW': (98, 4),
    '13|BMR': (124, 16),
    '14|HKS': (156, 25),
    '15|BMB': (197, 14),
    '16|MSR': (218, 6),
    '17|POW': (237, 22),
    # itemtracker row 2
    '21|MNP': (55, 50),
    '22|FIR': (86, 53),
    '23|ICR': (121, 53),
    '24|BBS': (148, 46),
    '25|ETH': (187, 55),
    '26|QUK': (228, 55),
    # itemtracker row 3
    '31|EP': (15, 72),
    '32|LMP': (89, 85),
    '33|HMR': (122, 78),
    '34|SVL': (148, 70),
    '35|FLU': (164, 89),
    '36|BGN': (190, 73),
    '37|BOK': (229, 92),
    '38|SLD': (55, 50),
    # itemtracker row 4
    '41|DP': (7, 119),
    '42|BTL': (89, 113),
    '43|SOM': (122, 120),
    '44|BYR': (157, 120),
    '45|CAP': (188, 118),
    '46|MIR': (227, 106),
    # itemtracker row 5
    '51|TH': (9, 156),
    '52|BTS': (80, 154),
    '53|GLV': (114, 150),
    '54|FLP': (163, 143),
    '55|MAG': (183, 158),
    '56|AG1': (219, 132),
    '57|AG2': (236, 151),
    # itemtracker row 6
    '61|POD': (16, 182),
    '62|SP': (50, 174),
    '63|SW': (82, 176),
    '64|TT': (115, 183),
    '65|IP': (155, 180),
    '66|MM': (195, 168),
    '67|TR': (233, 165),
    }

# default tracking point coordinates for the lightworld map tracker box
# absolute pixels relative to top left corner of box
DEFAULT_LIGHTWORLD_MAP_POINTS = {
    # lost woods & countryside
    "111|PED": (13, 7),
    "112|SHROOM": (31, 22),
    "113|HIDEOUT": (47, 33),
    "114|TREE": (77, 19),
    "115|FROG": (78, 138),
    "116|BAT": (82, 156),
    "117|BURIEDITEM": (74, 178),
    # kakariko
    "121|HUT": (34, 110),
    "122|WELL": (8, 110),
    "123|VENDOR": (24, 126),
    "124|CHICKEN": (22, 146),
    "125|KID": (40, 138),
    "126|TAVERN": (42, 156),
    "127|LIBRARY": (38, 178),
    "128|RACE": (8, 188),
    # south route and water checks
    "131|HOME": (140, 182),
    "132|DRAIN": (118, 252),
    "133|MMOLDCAVE": (164, 252),
    "134|IRCAVE": (226, 206),
    "135|LAKE": (182, 224),
    "136|WATERFALL": (228, 40),
    "137|ZORA": (244, 32),
    "138|ZORA_LEDGE": (242, 46),
    "139|BRIDGE": (180, 188),
    # eastern palace area
    "141|EP_DUNGEON": (230, 114),
    "142|EP_BOSS": (232, 108),
    "143|SAHASRALA": (206, 126),
    "144|SAHASCAVE": (206, 110),
    "145|WITCH": (206, 88),
    # desert palace area
    "151|DP_DUNGEON": (12, 220),
    "152|DP_BOSS": (14, 214),
    "153|DP_LEDGE": (6, 246),
    "154|DP_TABLET": (54, 250),
    "155|CHECKERBOARD": (44, 208),
    "156|AGINA": (50, 224),
    "157|CAVE45": (72, 228),
    # north route
    "161|BONK": (98, 80),
    "162|SANCTUARY": (116, 76),
    "163|GRAVE_LEDGE": (142, 74),
    "164|KINGSTOMB": (156, 80),
    # hyrule castle
    "171|AGA": (122, 148),
    "172|UNCLE": (152, 112),
    "173|CELLS": (126, 118),
    "174|DARKCROSS": (130, 104),
    "175|SEWERS": (136, 86),
    # mountain left
    "181|TH_DUNGEON": (148, 22),
    "182|TH_BOSS": (152, 16),
    "183|TH_TABLET": (106, 8),
    "184|OLDMAN": (106, 54),
    "185|SPECTCAVE": (124, 38),
    "186|SPECT_LEDGE": (130, 22),
    # mountain right
    "191|PARADOX": (210, 46),
    "192|SPIRAL": (204, 24),
    "193|FLOATING": (204, 8),
    "194|MIMIC": (216, 26),
    }

# default tracking point coordinates for the darkworld map tracker box
# absolute pixels relative to top left corner of box
DEFAULT_DARKWORLD_MAP_POINTS = {
    # skull woods & countryside
    "111|SW_DUNGEON": (5, 21),
    "112|SW_BOSS": (7, 16),
    "113|PURPLECHEST": (73, 141),
    "114|HAMMERPEGS": (77, 161),
    "115|STUMPY": (76, 184),
    # Dark Kakariko and Thieves Town
    "121|TT_DUNGEON": (22, 136),
    "122|TT_BOSS": (24, 130),
    "123|CHESTGAME": (8, 124),
    "124|BOMBHUT": (24, 156),
    "125|CHOUSE": (52, 128),
    "126|DIGGING": (12, 186),
    # swamp palace
    "131|SP_DUNGEON": (108, 254),
    "132|SP_BOSS": (110, 248),
    "133|HYPECAVE": (148, 206),
    # ice palace
    "141|IP_DUNGEON": (192, 238),
    "142|IP_BOSS": (194, 234),
    # palace of darkness
    "151|POD_DUNGEON": (226, 114),
    "152|POD_BOSS": (230, 110),
    # pyramid
    "161|PYRAMID_LEDGE": (144, 116),
    "162|REDBOMB": (116, 130),
    "163|CATFISH": (230, 46),
    # dark death mountain
    "171|TR_DUNGEON": (228, 26),
    "172|TR_BOSS": (230, 20),
    "173|GT_DUNGEON": (136, 22),
    "174|GT_BOSS": (137, 16),
    "175|SBUNNYCAVE": (214, 38),
    "176|HSCAVE": (208, 8),
    "177|HSCAVELOW": (208, 22),
    "178|SPIKECAVE": (142, 40),
    "179|BUMPERCAVE": (84, 40),
    # misery mire
    "181|MM_DUNGEON": (18, 232),
    "182|MM_BOSS": (20, 226),
    "183|MIRESHED": (5, 214),
    }