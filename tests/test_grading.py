from financebench_rag.grading import final_answer, numeric_match


def test_final_answer_is_read_from_the_last_line():
    assert final_answer("Step 1: add the margins\n**Final answer:** 10.3") == "10.3"
    assert final_answer("Final answer: 1\nFinal answer: 2") == "2"


def test_missing_final_answer_line_gives_none():
    assert final_answer("The margin is 10.3%") is None


def test_close_numbers_match():
    assert numeric_match("8.738", "$8.70")
    assert numeric_match("4.625", "$4.60")
    assert numeric_match("10.3", "10.3%")
    assert numeric_match("1,577", "$1577.00")


def test_far_numbers_and_refusals_do_not_match():
    assert not numeric_match("3.10", "$4.60")
    assert not numeric_match("not found", "$4.60")
