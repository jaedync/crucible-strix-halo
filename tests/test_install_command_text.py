"""install.sh command_text puts the quoted launcher path into each hint (r373 security review L2).

Bash 5.2 turned on patsub_replacement: an unquoted & in the replacement of ${var//pat/rep} means "the matched text". That
took the backslash that printf %q puts before & away, so a pasted hint ran a command in the background. The test path has
&, a space, ' and $. The hint must split back, as a shell reads it, to the exact launcher path."""
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def shell_words(text):
    """Words as a shell reads them; & ; | ( ) are their own words (punctuation_chars)."""
    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


class CommandTextTest(unittest.TestCase):
    def hint_words(self, folder):
        env = {**os.environ, "CRUCIBLE_DIR": folder, "HOME": tempfile.gettempdir()}
        r = subprocess.run(["bash", str(ROOT / "install.sh"), "--help"], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        line = next(ln for ln in r.stdout.splitlines() if "install Crucible and run" in ln)
        hint = line.split("install Crucible and run ", 1)[1]
        return shell_words(hint)

    def test_a_path_with_ampersand_space_quote_and_dollar_stays_one_word(self):
        for folder in ("/home/u/a&b", "/home/u/my dir", "/home/u/it's", "/home/u/$HOME", "/home/u/a&b c'd$e&&f",
                       "/home/u/plain"):
            with self.subTest(folder=folder):
                self.assertEqual(self.hint_words(folder), [folder + "/crucible", "setup"])


if __name__ == "__main__":
    unittest.main()
