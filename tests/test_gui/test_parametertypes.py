import spectrochempy.gui.widgets.parametertree as pt

def test_opts(app):

    paramSpec = [
        dict(name='bool', type='bool', readonly=True),
        dict(name='color', type='color', readonly=True),
    ]

    param = pt.Parameter.create(name='params', type='group', children=paramSpec)
    tree = pt.ParameterTree()
    tree.setParameters(param)

    assert list(param.param('bool').items.keys())[0].widget.isEnabled() is False
    assert list(param.param('color').items.keys())[0].widget.isEnabled() is False


    app.exec_()