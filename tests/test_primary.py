"""Primary-model contract checks; execute on Kaggle only."""
import torch

from wf.models import Net, build_model
from wf.primary import ARCHITECTURE, WildfireTD4, configuration, load_checkpoint


def test_selected_model_matches_legacy_checkpoint(tmp_path):
    old = Net({**ARCHITECTURE, 'cin': 33}, [0.]*33, [1.]*33).eval()
    assert sum(p.numel() for p in old.parameters()) == 5973133
    # EMA checkpoints omit integer BN counters, as the training runner does.
    state = {k: v for k, v in old.state_dict().items() if v.dtype.is_floating_point}
    path = tmp_path / 'best.pt'
    torch.save({'model': state, 'metrics': {'thr': .35}}, path)
    restored, _ = load_checkpoint(path)
    x = torch.randn(2, 33, 64, 64)
    with torch.no_grad():
        assert torch.equal(old(x)['main'], restored(x)['main'])
    assert isinstance(restored, WildfireTD4)


def test_recipe_and_factory_keep_selected_model():
    cfg = configuration('/kaggle/working/exploration/search_data')
    assert cfg['loss'] == {'name': 'focal_tversky', 'alpha': .4, 'beta': .6, 'gamma': 1.33}
    model = build_model({**cfg['model'], 'cin': 33}, [0.]*33, [1.]*33)
    assert isinstance(model, WildfireTD4)
    assert model.late is None
    assert cfg['train']['n_train'] is None
