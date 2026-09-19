from main import allocation_from_score


def test_allocation_rules():

    assert allocation_from_score(50) == 0.00
    assert allocation_from_score(65) == 0.40
    assert allocation_from_score(74.99) == 0.40
    assert allocation_from_score(75) == 0.60
    assert allocation_from_score(84.99) == 0.60
    assert allocation_from_score(85) == 0.80
    assert allocation_from_score(94.99) == 0.80
    assert allocation_from_score(95) == 1.00
    assert allocation_from_score(100) == 1.00


if __name__ == "__main__":

    test_allocation_rules()

    print("V9 allocation rules: OK")
