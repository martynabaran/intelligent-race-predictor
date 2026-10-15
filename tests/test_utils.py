from race_predictor.utils import timer


def test_timer_records_elapsed():
    with timer("x") as t:
        sum(range(1000))
    assert t.label == "x" and t.seconds > 0
