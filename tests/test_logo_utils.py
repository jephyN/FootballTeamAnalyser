"""
tests/test_logo_utils.py

Unit tests for logo_utils.py.

All tests are fully offline — urlopen is patched so no network calls are made.
"""

import io
from unittest.mock import MagicMock, patch


from logo_utils import _fetch_logo_pil, _wikimedia_thumbnail_url


# ---------------------------------------------------------------------------
# _wikimedia_thumbnail_url
# ---------------------------------------------------------------------------

class TestWikimediaThumbnailUrl:
    def test_en_svg_converted(self):
        url = (
            'https://upload.wikimedia.org/wikipedia/en/c/cf/LogoASMonacoFC2021.svg'
        )
        result = _wikimedia_thumbnail_url(url)
        assert result == (
            'https://en.wikipedia.org/w/thumb.php?f=LogoASMonacoFC2021.svg&w=320'
        )

    def test_commons_svg_converted(self):
        url = (
            'https://upload.wikimedia.org/wikipedia/commons/1/1b/FC_Bayern_logo.svg'
        )
        result = _wikimedia_thumbnail_url(url)
        assert result == (
            'https://commons.wikimedia.org/w/thumb.php?f=FC_Bayern_logo.svg&w=320'
        )

    def test_en_png_converted(self):
        url = (
            'https://upload.wikimedia.org/wikipedia/en/e/e9/Maccabi_Tel_Aviv.png'
        )
        result = _wikimedia_thumbnail_url(url)
        assert result == (
            'https://en.wikipedia.org/w/thumb.php?f=Maccabi_Tel_Aviv.png&w=320'
        )

    def test_thumb_variant_converted(self):
        # URL with /thumb/ path prefix (CDN thumbnail URL style)
        url = (
            'https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/'
            'LogoASMonacoFC2021.svg/200px-LogoASMonacoFC2021.svg.png'
        )
        result = _wikimedia_thumbnail_url(url)
        assert 'thumb.php' in result
        assert 'LogoASMonacoFC2021.svg' in result

    def test_custom_width_used(self):
        url = (
            'https://upload.wikimedia.org/wikipedia/en/c/cf/LogoASMonacoFC2021.svg'
        )
        result = _wikimedia_thumbnail_url(url, width=640)
        assert 'w=640' in result

    def test_none_url_returned_unchanged(self):
        assert _wikimedia_thumbnail_url(None) is None

    def test_empty_string_returned_unchanged(self):
        assert _wikimedia_thumbnail_url('') == ''

    def test_non_wikimedia_url_returned_unchanged(self):
        url = 'https://example.com/logo.png'
        assert _wikimedia_thumbnail_url(url) == url


# ---------------------------------------------------------------------------
# _fetch_logo_pil
# ---------------------------------------------------------------------------

def _make_png_bytes():
    """Return valid PNG bytes for a 1x1 red pixel image."""
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new('RGBA', (1, 1), color=(255, 0, 0, 255))
    img.save(buf, format='PNG')
    return buf.getvalue()


class TestFetchLogoPil:
    def _mock_response(self, data):
        resp = MagicMock()
        resp.read.return_value = data
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_returns_pil_image_on_success(self):
        png_bytes = _make_png_bytes()
        mock_resp = self._mock_response(png_bytes)
        with patch('logo_utils.urlopen', return_value=mock_resp):
            result = _fetch_logo_pil(
                'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg',
                size=(24, 24),
            )
        assert result is not None
        assert result.size == (24, 24)

    def test_returns_none_on_os_error(self):
        with patch('logo_utils.urlopen', side_effect=OSError('network error')):
            result = _fetch_logo_pil(
                'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg'
            )
        assert result is None

    def test_returns_none_on_value_error(self):
        with patch('logo_utils.urlopen', side_effect=ValueError('bad url')):
            result = _fetch_logo_pil(
                'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg'
            )
        assert result is None

    def test_retries_on_429(self):
        png_bytes = _make_png_bytes()
        mock_resp = self._mock_response(png_bytes)
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError('HTTP Error 429: Too Many Requests')
            return mock_resp

        with patch('logo_utils.urlopen', side_effect=side_effect):
            with patch('logo_utils.time.sleep'):   # skip real sleep
                result = _fetch_logo_pil(
                    'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg'
                )
        assert result is not None
        assert call_count[0] == 2

    def test_returns_none_after_three_429_failures(self):
        with patch('logo_utils.urlopen', side_effect=OSError('HTTP Error 429')):
            with patch('logo_utils.time.sleep'):
                result = _fetch_logo_pil(
                    'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg'
                )
        assert result is None

    def test_image_mode_converted_to_rgba(self):
        png_bytes = _make_png_bytes()
        mock_resp = self._mock_response(png_bytes)
        with patch('logo_utils.urlopen', return_value=mock_resp):
            result = _fetch_logo_pil(
                'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg',
                size=(32, 32),
            )
        assert result.mode == 'RGBA'

    def test_image_resized_to_requested_size(self):
        png_bytes = _make_png_bytes()
        mock_resp = self._mock_response(png_bytes)
        with patch('logo_utils.urlopen', return_value=mock_resp):
            result = _fetch_logo_pil(
                'https://upload.wikimedia.org/wikipedia/en/c/cf/Logo.svg',
                size=(48, 48),
            )
        assert result.size == (48, 48)
