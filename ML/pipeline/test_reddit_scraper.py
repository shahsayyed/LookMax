"""
Unit Tests for Reddit Playwright JSON Scraper (reddit_scraper.py)
=================================================================
Tests JSON parsing, URL construction (feeds & search endpoints), image extraction,
gallery handling, HTML entity unescaping, video/removed filtering, deduplication,
and category normalization.
"""

import json
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reddit_scraper import (
    RateLimitTracker,
    build_reddit_json_url,
    clean_and_unescape_url,
    extract_image_urls,
    is_deleted_or_removed,
    is_direct_image_url,
    is_video_or_media_embed,
    normalize_categories,
)


class TestRedditScraper(unittest.TestCase):

    def test_build_reddit_json_url_simple_subreddit(self):
        url = build_reddit_json_url("OUTFITS", listing="hot", limit=100)
        self.assertEqual(url, "https://www.reddit.com/r/OUTFITS/hot.json?limit=100")

    def test_build_reddit_json_url_with_prefix_and_after(self):
        url = build_reddit_json_url("r/streetwear", listing="top", after="t3_abc123", limit=50, time_period="year")
        self.assertIn("https://www.reddit.com/r/streetwear/top.json?", url)
        self.assertIn("limit=50", url)
        self.assertIn("after=t3_abc123", url)
        self.assertIn("t=year", url)

    def test_build_reddit_json_url_search_in_subreddit(self):
        url = build_reddit_json_url(
            target="malefashionadvice",
            query="tailored suit",
            sort="top",
            time_period="all",
            limit=100,
        )
        self.assertIn("https://www.reddit.com/r/malefashionadvice/search.json?", url)
        self.assertIn("q=tailored+suit", url)
        self.assertIn("restrict_sr=1", url)
        self.assertIn("sort=top", url)
        self.assertIn("t=all", url)
        self.assertIn("limit=100", url)

    def test_build_reddit_json_url_global_search(self):
        url = build_reddit_json_url(
            target="all",
            query="standing posture full body",
            sort="top",
            limit=50,
            after="t3_page2",
        )
        self.assertIn("https://www.reddit.com/search.json?", url)
        self.assertIn("q=standing+posture+full+body", url)
        self.assertIn("type=link", url)
        self.assertIn("after=t3_page2", url)
        self.assertIn("limit=50", url)

    def test_build_reddit_json_url_full_url(self):
        custom_url = "https://www.reddit.com/r/Posture/top"
        url = build_reddit_json_url(custom_url, after="t3_xyz789", limit=25, time_period="all")
        self.assertIn("https://www.reddit.com/r/Posture/top.json", url)
        self.assertIn("limit=25", url)
        self.assertIn("after=t3_xyz789", url)
        self.assertIn("t=all", url)

    def test_normalize_categories(self):
        # 1. List of subreddit strings
        cats_str = ["OUTFITS", "r/streetwear"]
        norm1 = normalize_categories(cats_str)
        self.assertEqual(len(norm1), 2)
        self.assertEqual(norm1[0]["category"], "reddit_OUTFITS")
        self.assertEqual(norm1[1]["category"], "reddit_streetwear")

        # 2. List of dicts (from config or queries file)
        cats_dict = [
            {"category": "men_u35_suit", "subreddit": "malefashionadvice", "query": "tailored suit"},
            {"folder": "reddit_outfits", "subreddit": "OUTFITS"},
        ]
        norm2 = normalize_categories(cats_dict)
        self.assertEqual(len(norm2), 2)
        self.assertEqual(norm2[0]["category"], "men_u35_suit")
        self.assertEqual(norm2[0]["query"], "tailored suit")
        self.assertEqual(norm2[1]["category"], "reddit_outfits")

    def test_rate_limit_tracker_defaults(self):
        tracker = RateLimitTracker(
            delay_range=(3.5, 7.0),
            batch_size=10,
            batch_cooldown=20.0,
            category_cooldown=8.0,
        )
        self.assertEqual(tracker.delay_range, (3.5, 7.0))
        self.assertEqual(tracker.batch_size, 10)
        self.assertEqual(tracker.batch_cooldown, 20.0)
        self.assertEqual(tracker.category_cooldown, 8.0)
        self.assertEqual(tracker.request_count, 0)

    def test_clean_and_unescape_url(self):
        raw = "https://preview.redd.it/test.jpg?width=1080&amp;crop=smart&amp;auto=webp&amp;s=abc"
        cleaned = clean_and_unescape_url(raw)
        self.assertEqual(cleaned, "https://preview.redd.it/test.jpg?width=1080&crop=smart&auto=webp&s=abc")
        self.assertNotIn("&amp;", cleaned)

    def test_is_direct_image_url(self):
        self.assertTrue(is_direct_image_url("https://i.redd.it/sample123.jpg"))
        self.assertTrue(is_direct_image_url("https://i.imgur.com/sample123.png"))
        self.assertTrue(is_direct_image_url("https://example.com/photo.jpeg?param=123"))
        self.assertTrue(is_direct_image_url("https://example.com/photo.webp"))
        self.assertFalse(is_direct_image_url("https://www.reddit.com/r/OUTFITS/comments/123/my_fit/"))
        self.assertFalse(is_direct_image_url("https://v.redd.it/sample123"))

    def test_is_video_or_media_embed(self):
        self.assertTrue(is_video_or_media_embed({"is_video": True}))
        self.assertTrue(is_video_or_media_embed({"post_hint": "hosted:video"}))
        self.assertTrue(is_video_or_media_embed({"domain": "v.redd.it"}))
        self.assertTrue(is_video_or_media_embed({"domain": "youtube.com"}))
        self.assertFalse(is_video_or_media_embed({"domain": "i.redd.it", "is_video": False}))

    def test_is_deleted_or_removed(self):
        self.assertTrue(is_deleted_or_removed({"removed_by_category": "moderator"}))
        self.assertTrue(is_deleted_or_removed({"selftext": "[removed]"}))
        self.assertTrue(is_deleted_or_removed({"title": "[deleted]"}))
        self.assertFalse(is_deleted_or_removed({"title": "Great outfit", "selftext": "Check this out"}))

    def test_extract_direct_image_urls(self):
        mock_payload = {
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "id": "post1",
                            "title": "Summer Outfit",
                            "url": "https://i.redd.it/img1.jpg",
                            "is_video": False,
                        },
                    },
                    {
                        "kind": "t3",
                        "data": {
                            "id": "post2",
                            "title": "Casual Style",
                            "url_overridden_by_dest": "https://i.imgur.com/img2.png",
                            "url": "https://reddit.com/r/OUTFITS/comments/post2",
                            "is_video": False,
                        },
                    },
                    {
                        "kind": "t3",
                        "data": {
                            "id": "post3_video",
                            "title": "Outfit Video Walkthrough",
                            "url": "https://v.redd.it/vid123",
                            "is_video": True,
                        },
                    },
                    {
                        "kind": "t3",
                        "data": {
                            "id": "post4_removed",
                            "title": "[deleted]",
                            "url": "https://i.redd.it/deleted.jpg",
                            "removed_by_category": "deleted",
                        },
                    },
                ]
            }
        }

        urls = extract_image_urls(mock_payload)
        self.assertEqual(len(urls), 2)
        self.assertEqual(urls[0], "https://i.redd.it/img1.jpg")
        self.assertEqual(urls[1], "https://i.imgur.com/img2.png")

    def test_extract_gallery_urls_with_unescape(self):
        mock_gallery_payload = {
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "id": "gallery_post_1",
                            "title": "Lookbook 3 Outfits",
                            "is_gallery": True,
                            "gallery_data": {
                                "items": [
                                    {"media_id": "media_item_1"},
                                    {"media_id": "media_item_2"},
                                ]
                            },
                            "media_metadata": {
                                "media_item_1": {
                                    "status": "valid",
                                    "s": {
                                        "u": "https://preview.redd.it/item1.jpg?width=1080&amp;format=pjpg&amp;auto=webp&amp;s=abc",
                                        "x": 1080,
                                        "y": 1440,
                                    },
                                },
                                "media_item_2": {
                                    "status": "valid",
                                    "s": {
                                        "u": "https://preview.redd.it/item2.png?width=1080&amp;format=png&amp;auto=webp&amp;s=def",
                                        "x": 1080,
                                        "y": 1440,
                                    },
                                },
                                "media_item_invalid": {
                                    "status": "failed",
                                },
                            },
                        },
                    }
                ]
            }
        }

        urls = extract_image_urls(mock_gallery_payload)
        self.assertEqual(len(urls), 2)
        self.assertEqual(
            urls[0],
            "https://preview.redd.it/item1.jpg?width=1080&format=pjpg&auto=webp&s=abc",
        )
        self.assertEqual(
            urls[1],
            "https://preview.redd.it/item2.png?width=1080&format=png&auto=webp&s=def",
        )
        self.assertNotIn("&amp;", urls[0])
        self.assertNotIn("&amp;", urls[1])

    def test_extract_fallback_preview_image(self):
        mock_preview_payload = {
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "id": "post_external_link",
                            "title": "Article with thumbnail",
                            "url": "https://fashionblog.com/article/123",
                            "preview": {
                                "images": [
                                    {
                                        "source": {
                                            "url": "https://external-preview.redd.it/thumb.jpg?width=960&amp;auto=webp&amp;s=12345",
                                            "width": 960,
                                            "height": 720,
                                        }
                                    }
                                ]
                            },
                        },
                    }
                ]
            }
        }

        urls = extract_image_urls(mock_preview_payload)
        self.assertEqual(len(urls), 1)
        self.assertEqual(
            urls[0],
            "https://external-preview.redd.it/thumb.jpg?width=960&auto=webp&s=12345",
        )
        self.assertNotIn("&amp;", urls[0])

    def test_extract_deduplication(self):
        mock_duplicate_payload = {
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "id": "post_a",
                            "url": "https://i.redd.it/same_image.jpg",
                        },
                    },
                    {
                        "kind": "t3",
                        "data": {
                            "id": "post_b",
                            "url": "https://i.redd.it/same_image.jpg",
                        },
                    },
                ]
            }
        }
        urls = extract_image_urls(mock_duplicate_payload)
        self.assertEqual(len(urls), 1)
        self.assertEqual(urls[0], "https://i.redd.it/same_image.jpg")


if __name__ == "__main__":
    unittest.main()
