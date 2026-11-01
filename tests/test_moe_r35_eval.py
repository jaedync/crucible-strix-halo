"""CPU checks of bench/moe_r35_eval.py (moe-r35): the bench line parser, the bytes verdict and the PICK rule."""
import importlib.util
import pathlib
import tempfile
import unittest

PATH = pathlib.Path(__file__).resolve().parents[1] / "bench/moe_r35_eval.py"
spec = importlib.util.spec_from_file_location("moe_r35_eval", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def line(r, v, us, status="clean", mb=None):
    mb = mb if mb is not None else 3.7 + 2.5 * (6 + 5 * r)
    return (f"R={r} v{v} med {us:.1f} us p10 {us - 1:.1f} p90 {us + 1:.1f} | {mb:.2f} MB {mb / us * 1e3:.1f} GB/s | "
            f"H bitwise equal (0 of 4 calls differ, NaN 0) | xh stray 0 calls (first word 0) | partial stray 0 calls "
            f"(first word 0) | ctl not reset 0 calls | {status}")


def run(lines):
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "p1.txt").write_text("\n".join(lines) + "\nSUMMARY PASS\n")
        return module.report([d])


def table(fast):
    """v170 at 70 us + bytes / 245.5 GB/s; v181 `fast` us faster at every R; v180 slower at R 4."""
    out = []
    for r in range(1, 9):
        base = 70 + (3.7 + 2.5 * (6 + 5 * r)) / 245.5 * 1e3
        out += [line(r, 170, base), line(r, 181, base - fast), line(r, 180, base + (5 if r == 4 else -5)),
                line(r, 186, base, "mutant caught")]
    return out


class EvalTests(unittest.TestCase):
    def test_pick_and_fit(self):
        text = run(table(10.0))
        self.assertIn("BYTES PASS", text)
        last = text.splitlines()[-1]
        self.assertTrue(last.startswith("PICK 5 v181 saving 10.0 us"), last)
        fit = [x for x in text.splitlines() if x.startswith("| v170 |")][0]
        self.assertIn("| 70.0 | 245.5 |", fit)

    def test_no_pick_when_not_faster_at_r3_to_5(self):
        self.assertTrue(run(table(1.0)).splitlines()[-1].startswith("PICK none"))  # 1 us: ratio above 0.99

    def test_bytes_fail(self):
        for bad in ("FAIL", "MUTANT PASSED (blind check)"):
            text = run(table(10.0) + [line(4, 182, 300.0, bad)])
            self.assertIn("BYTES FAIL", text)


if __name__ == "__main__":
    unittest.main()


class CounterTests(unittest.TestCase):
    def test_summary(self):
        path = pathlib.Path(__file__).resolve().parents[1] / "bench/moe_r35_counters.py"
        sp = importlib.util.spec_from_file_location("moe_r35_counters", path)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        k = '"void crucible::MoeSgKernel<4, 2, 1, 4, 0, 12, 1, 1, 3>(crucible::LinJob const*, int)"'
        rows = ["Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value"]
        for d in (1, 2):  # two dimensions per counter are summed per dispatch
            rows += [f"{d},{k},SQ_WAVE_CYCLES,500", f"{d},{k},SQ_WAVE_CYCLES,500", f"{d},{k},SQ_WAIT_INST_ANY,100",
                     f"{d},{k},SQ_WAIT_INST_ANY,150", f'{d},"crucible::MoeFusedKernel<4>",SQ_WAVE_CYCLES,7']
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "set1").mkdir()
            (pathlib.Path(d) / "set1" / "run_counter_collection.csv").write_text("\n".join(rows) + "\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                mod.main(d)
        line = [x for x in buf.getvalue().splitlines() if x.startswith("| R4 D2 NP3 |")]
        self.assertEqual(line, ["| R4 D2 NP3 | 250 | 1000 | 25.0 |"])
