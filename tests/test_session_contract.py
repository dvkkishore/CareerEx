"""Offline contract tests: do not import the database-backed app or access MySQL."""
import ast
import csv
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename="app.py")


class CareerExOfflineContracts(unittest.TestCase):
    def test_login_state_uses_request_session(self):
        declarations = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "is_logged_in"]
        self.assertEqual(len(declarations), 1)
        helper = compile(ast.Module(body=declarations, type_ignores=[]), "app.py", "exec")
        ctx = {"session": {}}
        exec(helper, ctx)
        self.assertFalse(ctx["is_logged_in"]())
        ctx["session"] = {"user_id": 21}
        self.assertTrue(ctx["is_logged_in"]())

    def test_no_global_login_state_assignment(self):
        global_names = {
            target.id
            for n in TREE.body
            if isinstance(n, ast.Assign)
            for target in n.targets
            if isinstance(target, ast.Name)
        }
        self.assertFalse({"logged_in", "logout_success"} & global_names)

    def test_job_dataset_has_required_columns(self):
        with (ROOT / "FinalJob.csv").open(encoding="utf-8-sig", newline="") as handle:
            columns = set(next(csv.reader(handle)))
        self.assertTrue({"Role", "Skills", "Industry"} <= columns)
        self.assertIn("pd.read_csv('FinalJob.csv')", SOURCE)
        self.assertIn("jobs_df['Skills']", SOURCE)


if __name__ == "__main__":
    unittest.main()
