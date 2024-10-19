
# Define the predefined RGB values for each color label
COLOR_LABELS_MAP_TRACKERS = {
    'DEFAULT': {
        'RED': (230, 0, 0),
        'GREEN': (20, 255, 20),
        'LIGHTBLUE': (40, 180, 240),
        'DARKBLUE': (0, 0, 240),
        'ORANGE': (240, 160, 20),
        'YELLOW': (245, 255, 15),
        'GREY': (128, 128, 128),
        'GREY-2': (43, 134, 29),  # for Dunka Compact 2024; GREY-LIGHTBLUE
        'GREY-3': (68, 221, 65),  # for Dunka Compact 2024; GREY-GREEN
        'GREY-4': (201, 207, 59),  # for Dunka Compact 2024; GREY-YELLOW
        'GREY-5': (184, 42, 60),  # for Dunka Compact 2024; GREY-RED
        'GREY-6': (126, 44, 127),  # for Dunka Compact 2024; GREY-PURPLE
        'PURPLE': (128, 0, 128),
    },
}

# Define the predefined RGB values for each item tracker label
COLOR_LABELS_ITEM_TRACKER = {
    '11|SWO': {
        'NONE': (0, 0, 0),
        'FIGHTER': (248, 248, 248),
        'MASTER': (160, 248, 216),
        'TEMPERED-1': (248, 160, 40),
        'TEMPERED-2': (248, 248, 200),  # BUTTER
    },
    '12|SLD': {
        'NONE': (0, 0, 0),
        'AVLBL-1': (248, 248, 248),  # WOOD
        'AVLBL-2': (176, 40, 40),  # IRON
        'AVLBL-3': (200, 224, 224),  # MIRROR
    },
    '13|BOW': {
        'NONE': (0, 0, 0),
        'SIMPLE': (248, 176, 80),
        'SILVERS': (248, 248, 248),
    },
    '14|BMR': {  # backward compatibility; do not use!
        'NONE': (36, 42, 58),
        'BLUE': (144, 168, 232),
        'RED': (224, 112, 112),
    },
    '14|BLR': {
        'NONE': (36, 42, 58),
        'BLUE': (144, 168, 232),
        'RED': (224, 112, 112),
    },
    '15|RMR': {
        'NONE': (36, 42, 58),
        'BLUE': (144, 168, 232),
        'RED': (224, 112, 112),
    },
    '90|BMR': {
        'NONE': (36, 42, 58),
        'BLUE': (144, 168, 232),
        'RED': (224, 112, 112),
    },
    '16|HKS': {
        'NONE': (44, 10, 10),
        'AVLBL': (176, 40, 40),
    },
    '17|BMB': {
        'NONE': (64, 64, 64),
        'AVLBL': (255, 255, 255),
    },
    '18|MSR': {
        'NONE': (46, 34, 8),
        'AVLBL': (200, 48, 24),
    },
    '19|POW': {
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
        # 'NONE': (62, 62, 62),
        'NONE': (0, 0, 0),
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
    "70|EPP": {
        # 'NONE': (255, 255, 255),
        'NONE': (0, 0, 0),
        # 'CRYSTAL': (224, 112, 112),
        'CRYSTAL': (100, 100, 100),
        # 'CRYSTAL': (40, 180, 240),
        # 'CRYSTAL_BLUE': (0, 0, 240),
        # 'PENDANT_RB': (249, 177, 79),
        # 'PENDANT_GREEN': (249, 209, 54),
    },
    "71|DPP": {
        # 'NONE': (255, 255, 255),
        'NONE': (0, 0, 0),
        # 'CRYSTAL': (224, 112, 112),
        'CRYSTAL': (100, 100, 100),
        # 'CRYSTAL': (40, 180, 240),
        # 'CRYSTAL_BLUE': (0, 0, 240),
        # 'PENDANT_RB': (249, 177, 79),
        # 'PENDANT_GREEN': (249, 209, 54),
    },
    "72|THP": {
        # 'NONE': (255, 255, 255),
        'NONE': (0, 0, 0),
        # 'CRYSTAL': (224, 112, 112),
        'CRYSTAL': (100, 100, 100),
        # 'CRYSTAL': (40, 180, 240),
        # 'CRYSTAL_BLUE': (0, 0, 240),
        # 'PENDANT_RB': (249, 177, 79),
        # 'PENDANT_GREEN': (249, 209, 54),
    },
    "73|PODP": {
        # 'NONE': (255, 255, 255),
        'NONE': (0, 0, 0),
        # 'CRYSTAL': (224, 112, 112),
        'CRYSTAL': (100, 100, 100),
        # 'CRYSTAL': (40, 180, 240),
        # 'CRYSTAL_BLUE': (0, 0, 240),
        # 'PENDANT_RB': (249, 177, 79),
        # 'PENDANT_GREEN': (249, 209, 54),
    },
    "74|SPP": {
        # 'NONE': (255, 255, 255),
        'NONE': (0, 0, 0),
        # 'CRYSTAL': (224, 112, 112),
        'CRYSTAL': (100, 100, 100),
        # 'CRYSTAL': (40, 180, 240),
        # 'CRYSTAL_BLUE': (0, 0, 240),
        # 'PENDANT_RB': (249, 177, 79),
        # 'PENDANT_GREEN': (249, 209, 54),
    },
    "75|SWP": {
        # 'NONE': (255, 255, 255),
        'NONE': (0, 0, 0),
        # 'CRYSTAL': (224, 112, 112),
        'CRYSTAL': (100, 100, 100),
        # 'CRYSTAL': (40, 180, 240),
        # 'CRYSTAL_BLUE': (0, 0, 240),
        # 'PENDANT_RB': (249, 177, 79),
        # 'PENDANT_GREEN': (249, 209, 54),
    },
    "76|TTP": {
        # 'NONE': (255, 255, 255),
        'NONE': (0, 0, 0),
        # 'CRYSTAL': (224, 112, 112),
        'CRYSTAL': (100, 100, 100),
        # 'CRYSTAL': (40, 180, 240),
        # 'CRYSTAL_BLUE': (0, 0, 240),
        # 'PENDANT_RB': (249, 177, 79),
        # 'PENDANT_GREEN': (249, 209, 54),
    },
    "77|IPP": {
        # 'NONE': (255, 255, 255),
        'NONE': (0, 0, 0),
        # 'CRYSTAL': (224, 112, 112),
        'CRYSTAL': (100, 100, 100),
        # 'CRYSTAL': (40, 180, 240),
        # 'CRYSTAL_BLUE': (0, 0, 240),
        # 'PENDANT_RB': (249, 177, 79),
        # 'PENDANT_GREEN': (249, 209, 54),
    },
    "78|MMP": {
        # 'NONE': (255, 255, 255),
        'NONE': (0, 0, 0),
        # 'CRYSTAL': (224, 112, 112),
        'CRYSTAL': (100, 100, 100),
        # 'CRYSTAL': (40, 180, 240),
        # 'CRYSTAL_BLUE': (0, 0, 240),
        # 'PENDANT_RB': (249, 177, 79),
        # 'PENDANT_GREEN': (249, 209, 54),
    },
    "79|TRP": {
        # 'NONE': (255, 255, 255),
        'NONE': (0, 0, 0),
        # 'CRYSTAL': (224, 112, 112),
        'CRYSTAL': (100, 100, 100),
        # 'CRYSTAL': (40, 180, 240),
        # 'CRYSTAL_BLUE': (0, 0, 240),
        # 'PENDANT_RB': (249, 177, 79),
        # 'PENDANT_GREEN': (249, 209, 54),
    },
    'DEFAULT': {
        'OFF': (62, 62, 62),
        'ON': (248, 248, 248),
    }
}
