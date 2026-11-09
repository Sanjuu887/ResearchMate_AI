import unittest

from report_exports import make_report_pdf


class ReportPdfTests(unittest.TestCase):
    def test_pdf_export_contains_a_valid_pdf_document(self):
        result = {
            "answer": (
                "# Executive summary\n"
                "A source-supported answer [1].\n\n"
                "| Claim | Evidence |\n"
                "| --- | --- |\n"
                "| Result | Confirmed |"
            ),
            "depth": "Deep Research",
            "generated_at": "2025-01-01T12:00:00+00:00",
            "sources": [
                {
                    "id": "S1",
                    "title": "Example & Study",
                    "url": "https://example.com/report",
                    "retrieved_as": "page",
                    "published_date": "2025-01-01",
                    "content": "Supporting excerpt.",
                }
            ],
        }

        document = make_report_pdf("Does the finding hold?", result)

        self.assertTrue(document.startswith(b"%PDF-"))
        self.assertTrue(document.rstrip().endswith(b"%%EOF"))
        self.assertGreater(len(document), 1000)


if __name__ == "__main__":
    unittest.main()
