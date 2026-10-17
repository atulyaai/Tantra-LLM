import pytest
from Tantra.benchmark_training import choose


def test_choose_fastest_safe_microbatch():
    assert choose([{'microbatch':2,'safe':True,'tokens_per_second':400},
                   {'microbatch':8,'safe':True,'tokens_per_second':350},
                   {'microbatch':16,'safe':False,'tokens_per_second':900}])==2
    with pytest.raises(MemoryError):choose([{'microbatch':16,'safe':False}])
