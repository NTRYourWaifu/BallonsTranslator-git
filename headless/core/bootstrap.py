"""
headless bootstrap：把 BallonsTranslator 的三個模組（yolov8 偵測 / llm_ocr / lama inpaint）
從 config.json 讀出設定、建構好，供無頭 pipeline 直接呼叫。

設計原則（見 docs/翻譯器重做_需求與實作計畫.md D10）：
  - 整份讀 BT config/config.json，GUI 調過什麼就用什麼，不自訂、不做單點補丁。
  - 但兩個 headless 專屬覆寫（override_params）：
      1. llm_ocr.save_grid_debug = False（零中間檔）
      2. OCR 模組改用 headless 專屬 fork（core/ocr_llm_headless.py，複製自 modules/ocr/ocr_llm.py、
         不動 GUI 版本），三層 fallback 模型：
           model=gemini-3.1-flash-lite（主）
           timeout_fallback_model=gemini-3-flash-preview（逾時/伺服器忙碌類，Plan A2）
           safety_model=gemini-2.5-flash（安全過濾擋住類，Plan A2 + Plan B 切片救援）
           fallback_model=grok（最終備援，Plan C）
         見 docs/翻譯器待辦04_模型fallback策略與卡住診斷.md。
"""
import os
import sys

# offscreen 必須在 import 任何 qt 之前設定
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

BT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_bt_path():
    """chdir 到 BT root 並把它加進 sys.path（模型路徑相對 data/models/）。"""
    os.chdir(BT_ROOT)
    if BT_ROOT not in sys.path:
        sys.path.insert(0, BT_ROOT)


def _init_fonts(verbose=True):
    """搬 launch.py 的字型初始化：offscreen 下 Windows 必須手動載系統字型（issue #519）。"""
    from qtpy.QtGui import QFontDatabase, QFont, QGuiApplication
    from qtpy.QtCore import QStandardPaths
    import utils.shared as shared
    from utils.io_utils import find_all_files_recursive

    # PATH_FONTS/FONT_EXTS 是 launch.py 本地常數，不在 shared，這裡自定義
    PATH_FONTS = os.path.join(BT_ROOT, 'fonts')
    FONT_EXTS = {'.ttf', '.otf', '.ttc', '.pfb'}

    # 1) 載自訂字型 fonts/
    try:
        if os.path.exists(PATH_FONTS):
            for fp in find_all_files_recursive(PATH_FONTS, FONT_EXTS):
                idx = QFontDatabase.addApplicationFont(fp)
                if idx >= 0:
                    fams = QFontDatabase.applicationFontFamilies(idx)
                    if fams:
                        shared.CUSTOM_FONTS.append(fams[0])
    except Exception as e:
        if verbose:
            print(f'[bootstrap][font] 載 fonts/ 失敗（非致命）：{e}')

    # 2) offscreen/Windows 手動掃系統字型目錄（關鍵修復，issue #519）
    if sys.platform == 'win32':
        try:
            font_dirs = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.FontsLocation)
        except Exception:
            font_dirs = QStandardPaths.standardLocations(QStandardPaths.FontsLocation)
        loaded = 0
        for fd in font_dirs:
            try:
                for fp in find_all_files_recursive(fd, FONT_EXTS):
                    if QFontDatabase.addApplicationFont(fp) >= 0:
                        loaded += 1
            except Exception:
                continue
        if verbose:
            print(f'[bootstrap][font] 系統字型載入 {loaded} 個檔')

    # 3) 建 FONT_FAMILIES 集合（供 resolve 字型用）
    try:
        shared.FONT_FAMILIES = set(f for f in QFontDatabase.families())
    except Exception:
        try:
            shared.FONT_FAMILIES = set(QFontDatabase().families())
        except Exception:
            shared.FONT_FAMILIES = set()

    # 4) 預設字型
    yahei = QFont('Microsoft YaHei UI')
    if yahei.exactMatch():
        QGuiApplication.setFont(yahei)
        shared.DEFAULT_FONT_FAMILY = 'Microsoft YaHei UI'
        shared.APP_DEFAULT_FONT = 'Microsoft YaHei UI'

    # 5) ⚠️字型替換修正（offscreen 關鍵 bug）：
    #    config 的 font_family 可能是「family + 字重後綴」如 'Yu Gothic UI Semibold'，
    #    但 offscreen 字型庫只認基礎 family 'Yu Gothic UI'（Semibold 是 weight 變體不是 family 名）。
    #    直接 QFont('Yu Gothic UI Semibold') 會 fallback 到亂七八糟的字型（實測 Agency FB）。
    #    用 QFont.insertSubstitution 告訴 Qt：找不到這個 family 就用砍掉字重後綴的基礎 family。
    _register_font_substitutions(shared, verbose)

    if verbose:
        _gf = _config_font_family()
        _test = QFont(_gf); _test.setPixelSize(40)
        from qtpy.QtGui import QFontInfo
        print(f'[bootstrap][font] FONT_FAMILIES={len(shared.FONT_FAMILIES)} 個  '
              f'config字型={_gf!r}→實際={QFontInfo(_test).family()!r}  YaHei match={yahei.exactMatch()}')


# 常見字重後綴（Qt family 名不含這些，它們是 weight 變體）
_WEIGHT_SUFFIXES = (
    ' Semibold', ' SemiBold', ' Demibold', ' DemiBold', ' Bold', ' Medium',
    ' Light', ' Regular', ' Black', ' Heavy', ' Thin', ' ExtraLight', ' ExtraBold',
)


def _config_font_family() -> str:
    """讀 config global_fontformat 的 font_family（headless 渲染實際會用的字型）。"""
    import utils.config as _cfgmod
    gf = getattr(_cfgmod.pcfg, 'global_fontformat', None)
    if isinstance(gf, dict):
        return gf.get('font_family') or 'Yu Gothic UI'
    fam = getattr(gf, 'font_family', None)
    return fam or 'Yu Gothic UI'


def _base_family(name: str) -> str:
    """砍掉字重後綴 → 基礎 family。'Yu Gothic UI Semibold' → 'Yu Gothic UI'。"""
    for suf in _WEIGHT_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _register_font_substitutions(shared, verbose):
    """把 config 用到、但 offscreen 找不到的帶字重 family 名，替換成基礎 family。"""
    from qtpy.QtGui import QFont
    fam = _config_font_family()
    if fam and fam not in shared.FONT_FAMILIES:
        base = _base_family(fam)
        if base != fam and base in shared.FONT_FAMILIES:
            QFont.insertSubstitution(fam, base)
            if verbose:
                print(f'[bootstrap][font] 字型替換：{fam!r} → {base!r}（offscreen 無此 family）')


class HeadlessModules:
    """載好的三模組容器。"""
    def __init__(self, detector, ocr, inpainter, cfg_module):
        self.detector = detector
        self.ocr = ocr
        self.inpainter = inpainter
        self.cfg = cfg_module   # pcfg.module，供讀 font_size_scale 等


def load_modules(*, only_gemini: bool = True, gemini_model: str = None,
                 verbose: bool = True) -> HeadlessModules:
    """
    讀 config → 建三模組。
    only_gemini=True：清空 llm_ocr 的 fallback_model / safety_model（不走 Grok）。
    gemini_model：若指定，覆寫主 model（例 'gemini-3.1-flash-lite'）。
    """
    ensure_bt_path()

    # QApplication 要先起（TextBlkItem/QTextDocument 渲染需要，且模組 import 可能碰 qt）
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    # ⚠️ 關鍵：offscreen 下 Windows 字型庫不會自動初始化（BT issue #519），
    # 不補這段中文/日文字型會渲染成 tofu 方塊 ■。搬 launch.py:242-274 的字型初始化。
    _init_fonts(verbose=verbose)

    from utils.config import load_config
    import utils.config as _cfgmod
    load_config()
    pcfg = _cfgmod.pcfg
    cfg_module = pcfg.module

    # 觸發三個 registry 註冊
    from modules.textdetector.base import TEXTDETECTORS
    from modules.ocr.base import OCR
    from modules.inpaint.base import INPAINTERS
    # ⚠️ package __init__ 只 import base，不會觸發具體實作的 @register 裝飾器。
    # 必須明確 import 具體模組檔，@register_* 才會把類別註冊進 module_dict。
    import modules.textdetector.detector_yolov8  # noqa  → register 'yolov8'
    try:
        from . import ocr_llm_headless  # noqa  → register 'llm_ocr_headless'（走 headless.core 套件時，如 cli.py 的正常呼叫路徑）
    except ImportError:
        # bootstrap.py 被直接當 __main__ 執行時（如煙窗測試）沒有 package context；
        # BT portable python 是嵌入式發行版，不會自動把腳本自身目錄加進 sys.path，這裡手動補上。
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        if _this_dir not in sys.path:
            sys.path.insert(0, _this_dir)
        import ocr_llm_headless  # noqa  → register 'llm_ocr_headless'
    # lama_large_512px / lama_mpe 註冊在 inpaint/base.py 本體（import base 即註冊）
    import modules.inpaint.base                  # noqa  → register 'lama_large_512px'

    det_name = cfg_module.textdetector
    ocr_name = cfg_module.ocr
    inp_name = cfg_module.inpainter
    if verbose:
        print(f'[bootstrap] textdetector={det_name} ocr={ocr_name} inpainter={inp_name}')

    def build(register, module_key, name, overrides=None):
        cls = register.module_dict[name]
        params = cfg_module.get_params(module_key).get(name)
        if params is not None:
            # get_params 回的是 {param: {'value':..., ...}} 結構，模組建構吃的就是這個
            if overrides:
                import copy
                params = copy.deepcopy(params)
                for k, v in overrides.items():
                    if k in params and isinstance(params[k], dict):
                        params[k]['value'] = v
                    else:
                        params[k] = {'value': v}
            return cls(**params)
        return cls()

    # ── OCR 覆寫 ──
    ocr_overrides = {'save_grid_debug': False}   # 零中間檔
    if only_gemini:
        # 三層 fallback（依錯誤類型分流，見 headless/core/ocr_llm_headless.py Plan A2）：
        # 逾時/伺服器忙碌 → timeout_fallback_model；安全過濾擋住 → safety_model；兩者都不行 → fallback_model（grok，最終）
        ocr_overrides['timeout_fallback_model'] = 'gemini-3-flash-preview'
        ocr_overrides['safety_model'] = 'gemini-2.5-flash'
        ocr_overrides['fallback_model'] = 'grok-4.20-0309-non-reasoning'
    if gemini_model:
        ocr_overrides['model'] = gemini_model

    detector = build(TEXTDETECTORS, 'textdetector', det_name)
    # OCR 用 headless 專屬 fork 的 class（'llm_ocr_headless'），固定讀 config.json 存的 'llm_ocr'
    # 參數當底（api key 等）——不用 cfg_module.ocr（GUI 目前選的引擎，可能被使用者切成別的，
    # 例如 manga_ocr，那樣讀不到 gemini_api_key 會讓 headless 靜默無翻譯結果）。
    # headless 本來就固定用 yolov8/llm_ocr/lama，不隨 GUI 當下選的引擎變動。
    ocr_cls = OCR.module_dict['llm_ocr_headless']
    ocr_base_params = cfg_module.get_params('ocr').get('llm_ocr')
    if ocr_base_params is not None:
        import copy
        ocr_params = copy.deepcopy(ocr_base_params)
        for k, v in ocr_overrides.items():
            if k in ocr_params and isinstance(ocr_params[k], dict):
                ocr_params[k]['value'] = v
            else:
                ocr_params[k] = {'value': v}
        ocr = ocr_cls(**ocr_params)
    else:
        ocr = ocr_cls()
    inpainter = build(INPAINTERS, 'inpainter', inp_name)

    # 預載模型（不走 load_model_on_demand，headless 一次跑到底、先載好省得每頁判斷）
    for m, tag in [(detector, 'detector'), (ocr, 'ocr'), (inpainter, 'inpainter')]:
        try:
            if not m.all_model_loaded():
                m.load_model()
                if verbose:
                    print(f'[bootstrap] {tag} 模型載入完成')
        except Exception as e:
            print(f'[bootstrap][WARN] {tag} load_model 失敗（可能 on-demand）：{e}')

    if verbose:
        print(f'[bootstrap] ocr 主模型={ocr.params["model"]["value"]!r} '
              f'timeout_fallback={ocr.params["timeout_fallback_model"]["value"]!r} '
              f'safety={ocr.params["safety_model"]["value"]!r} '
              f'fallback={ocr.params["fallback_model"]["value"]!r} '
              f'gemini_key_len={len(ocr.params["gemini_api_key"]["value"])} '
              f'grok_key_len={len(ocr.params["grok_api_key"]["value"])}')

    return HeadlessModules(detector, ocr, inpainter, cfg_module)


if __name__ == '__main__':
    # 煙霧測試：只載模組、不翻譯
    print('=== bootstrap 載模組煙霧測試 ===')
    mods = load_modules(only_gemini=True, gemini_model='gemini-3.1-flash-lite')
    print()
    print('[OK] 三模組載入成功：')
    print(f'  detector   = {type(mods.detector).__name__}')
    print(f'  ocr        = {type(mods.ocr).__name__}')
    print(f'  inpainter  = {type(mods.inpainter).__name__}')
