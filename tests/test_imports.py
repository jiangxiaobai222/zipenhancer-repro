def test_public_imports():
    from zipenhancer_repro.models.backbone import build_backbone

    model = build_backbone()
    assert model is not None
