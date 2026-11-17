from server import __main__ as startup


def test_ngram_gpu_warning_when_unset_or_zero():
    expected = ('WARNING: the n-gram table uses GPU memory because CRUCIBLE_PLE_HOST is not 1. '
                'This can leave too little memory for the model and its sessions. '
                'Set CRUCIBLE_PLE_HOST=1 before you start the server to keep the table on the host.')
    assert startup.ngram_notice({}) == expected
    assert startup.ngram_notice({'CRUCIBLE_PLE_HOST': '0'}) == expected
    assert startup.ngram_notice({'CRUCIBLE_PLE_HOST': '1'}) is None
