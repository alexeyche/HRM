from dataset.grammar import get_cfg, sample_programs, parse_program


def test_generates_parseable_programs():
    cfg = get_cfg()
    programs = sample_programs(cfg, n=1000)
    for program in programs:
        assert parse_program(program), f"Program {program} is not parseable"



