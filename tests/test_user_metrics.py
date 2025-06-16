import importlib.util
import pathlib
import sys
import types

import pytest

# Stub heavy external dependencies to simplify import
modules_to_stub = {
    'pandas': types.ModuleType('pandas'),
    'numpy': types.ModuleType('numpy'),
    'plotly': types.ModuleType('plotly'),
    'plotly.express': types.ModuleType('plotly.express'),
    'plotly.io': types.ModuleType('plotly.io'),
    'sklearn': types.ModuleType('sklearn'),
    'sklearn.preprocessing': types.ModuleType('sklearn.preprocessing'),
    'sklearn.cluster': types.ModuleType('sklearn.cluster'),
    'sklearn.metrics': types.ModuleType('sklearn.metrics'),
    'sklearn.metrics.pairwise': types.ModuleType('sklearn.metrics.pairwise'),
    'sklearn_extra': types.ModuleType('sklearn_extra'),
    'sklearn_extra.cluster': types.ModuleType('sklearn_extra.cluster'),
    'umap': types.ModuleType('umap'),
    'umap.umap_': types.ModuleType('umap.umap_'),
    'dateutil': types.ModuleType('dateutil'),
    'dateutil.relativedelta': types.ModuleType('dateutil.relativedelta'),
    'tqdm': types.ModuleType('tqdm'),
    'kneed': types.ModuleType('kneed'),
    'requests': types.ModuleType('requests'),
    'FinanceDataReader': types.ModuleType('FinanceDataReader'),
}

for name, module in modules_to_stub.items():
    if name not in sys.modules:
        sys.modules[name] = module
    if '.' in name:
        parent = name.split('.')[0]
        if parent not in sys.modules:
            sys.modules[parent] = types.ModuleType(parent)

# Add required attributes to stubs
sys.modules['pandas'].DataFrame = object
sys.modules['numpy'].array = lambda *a, **k: []
sys.modules['numpy'].nan = float('nan')
sys.modules['numpy'].inf = float('inf')
sys.modules['numpy'].sqrt = lambda x: 0
sys.modules['numpy'].abs = abs
sys.modules['sklearn.preprocessing'].RobustScaler = object
sys.modules['sklearn.cluster'].KMeans = object
sys.modules['sklearn.cluster'].DBSCAN = object
sys.modules['sklearn.metrics'].silhouette_score = lambda *a, **k: 0
sys.modules['sklearn.metrics.pairwise'].cosine_similarity = lambda *a, **k: 0
sys.modules['sklearn_extra.cluster'].KMedoids = object
sys.modules['umap.umap_'].UMAP = object
sys.modules['kneed'].KneeLocator = object
sys.modules['dateutil.relativedelta'].relativedelta = object
sys.modules['tqdm'].tqdm = lambda x, **k: x

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "etf_추천시스템.py"
spec = importlib.util.spec_from_file_location("etf_module", MODULE_PATH)
etf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(etf)

def test_derive_user_quantitative_indicators():
    user_profile = {
        'risk_tolerance': 4,
        'investment_horizon': 3,
        'goal': 5,
        'market_preference': 2,
        'experience': 3,
        'loss_aversion': 2,
        'theme_preference': 4,
    }
    indicators = etf.derive_user_quantitative_indicators(user_profile)
    assert indicators['risk_score'] == 4.0
    assert indicators['expected_return'] == 0.15
    assert indicators['market_scores'] == {'KR': 0.0, 'US': 1.0}
    assert indicators['user_theme_preference_code'] == 4
