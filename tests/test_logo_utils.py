"""
tests/test_logo_utils.py

Unit tests for logo_utils.py.

All tests are fully offline — urlopen is patched so no network calls are made.
"""

import io
from unittest.mock import MagicMock, patch

from PIL import Image

from logo_utils import _fetch_logo_pil, _wikimedia_thumbnail_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png_bytes():
    """Return valid PNG bytes for a 1×1 red RGBA pixel image."""
    buf = io.BytesIO()
    img = Image.new('RGBA', (1, 1), color=(255, 0, 0, 255))
    img.save(buf, format='PNG')
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _wikimedia_thumbnail_url
# ---------------------------------------------------------------------------

class TestWikimediaThumbnailUrl:
    """Tests for the Wikimedia upload URL → thumb.php conversion helper."""

    def test_en_svg_converted(self):
        """English-wiki SVG URL is rewritten to the thumb.php endpoint."""
        url = 'https://upload.wikimedia.org/wikipedia/en/c/cf/LogoASMonacoFC2021.svg'
        result = _wikimedia_thumbnail_url(url)
        assert result == (
            'https://en.wikipedia.org/w/thumb.php?f=LogoASMonacoFC2021.svg&w=320'
        )

    def test_commons_svg_converted(self):
        """Commons SVG URL maps to the commons.wikimedia.org host."""
        url = 'https://upload.wikimedia.org/wikipedia/commons/1/1b/FC_Bayern_logo.svg'
        result = _wikimedia_thumbnail_url(url)
        assert result == (
            'https://commons.wikimedia.org/w/thumb.php?f=FC_Bayern_logo.svg&w=320'
        )

    def test_en_png_converted(self):
        """English-wiki PNG URL is rewritten to the thumb.php endpoint."""
        url = 'https://upload.wikimedia.org/wikipedia/en/e/e9/Maccabi_Tel_Aviv.png'
        result = _wikimedia_thumbnail_url(url)
        assert result == (
            'https://en.wikipedia.org/w/thumb.php?f=Maccabi_Tel_Aviv.png&w=320'
        )

    def test_thumb_variant_converted(self):
        """CDN /thumb/ path variant is correctly parsed and converted."""
        url = (
            'https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/'
            'LogoASMonacoFC2021.svg/200px-LogoASMonacoFC2021.svg.png'
        )
        result = _wikimedia_thumbnail_url(url)
        assert 'thumb.php' in result
        assert 'LogoASMonacoFC2021.svg' in result

    def test_custom_width_used(self):
        """The width parameter is forwarded into the generated URL."""
        url = 'https://upload.wikimedia.org/wikipedia/en/c/cf/LogoASMonacoFC2021.svg'
        result = _wikimedia_thumbnail_url(url, width=640)
        assert 'w=640' in result

    def test_none_url_returned_unchanged(self):
        """None input is returned as-is without raising."""
        assert _wikimedia_thumbnail_url(None) is None

    def test_empty_string_returned_unchanged(self):
        """Empty string is returned unchanged."""
        assert _wikimedia_thumbnail_url('') == ''

    def test_non_wikimedia_url_returned_unchanged(self):
        """Non-Wikimedia URLs are returned without modification."""
        url = 'https://example.com/logo.png'
        assert _wikimedia_thumbnail_url(url) == url


# ---------------------------------------------------------------------------
# _fetch_logo_pil
# ---------------------------------------------------------------------------

class TestFetchLogoPil:
    """Tests for the PIL image download helper."""

    def _mock_response(self, data):
        """Return a context-manager mock that yields the given bytes."""
        resp = MagicMock()
        resp.read.return_value = data
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_returns_pil_image_on_success(self):
        """A valid response produces a PIL Image of the requested size."""
        mock_resp = self._mock_response(_make_png_bytes())
        with patch('logo_utils.urlopen', return_value=mock_resp):
            result = _fetch_logo_pil(
                'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg',
                size=(24, 24),
            )
        assert result is not None
        assert result.size == (24, 24)

    def test_returns_none_on_os_error(self):
        """A network OSError causes None to be returned."""
        with patch('logo_utils.urlopen', side_effect=OSError('network error')):
            result = _fetch_logo_pil(
                'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg'
            )
        assert result is None

    def test_returns_none_on_value_error(self):
        """A ValueError during fetch causes None to be returned."""
        with patch('logo_utils.urlopen', side_effect=ValueError('bad url')):
            result = _fetch_logo_pil(
                'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg'
            )
        assert result is None

    def test_retries_on_429(self):
        """A single HTTP 429 triggers a retry and ultimately succeeds."""
        mock_resp = self._mock_response(_make_png_bytes())
        call_count = [0]

        def side_effect(*_args, **_kwargs):
            """Fail on the first call, succeed on the second."""
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError('HTTP Error 429: Too Many Requests')
            return mock_resp

        with patch('logo_utils.urlopen', side_effect=side_effect):
            with patch('logo_utils.time.sleep'):
                result = _fetch_logo_pil(
                    'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg'
                )
        assert result is not None
        assert call_count[0] == 2

    def test_returns_none_after_three_429_failures(self):
        """Three consecutive HTTP 429 errors exhaust retries and return None."""
        with patch('logo_utils.urlopen', side_effect=OSError('HTTP Error 429')):
            with patch('logo_utils.time.sleep'):
                result = _fetch_logo_pil(
                    'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg'
                )
        assert result is None

    def test_image_mode_converted_to_rgba(self):
        """Downloaded image is always returned in RGBA mode."""
        mock_resp = self._mock_response(_make_png_bytes())
        with patch('logo_utils.urlopen', return_value=mock_resp):
            result = _fetch_logo_pil(
                'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg',
                size=(32, 32),
            )
        assert result.mode == 'RGBA'

    def test_image_resized_to_requested_size(self):
        """Downloaded image is resized to exactly the requested dimensions."""
        mock_resp = self._mock_response(_make_png_bytes())
        with patch('logo_utils.urlopen', return_value=mock_resp):
            result = _fetch_logo_pil(
                'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg',
                size=(48, 48),
            )
        assert result.size == (48, 48)
