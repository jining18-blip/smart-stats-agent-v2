import io, re, ssl, types
import requests
from requests.exceptions import SSLError

src = io.open('app.py', encoding='utf-8').read()
seg = src[src.index('def _kamis_ssl_context'):src.index('KAMIS_ITEM_CATEGORY_MAP = {')]

ns = {'_HAS_REQUESTS': True, '_requests': requests}
exec(compile(seg, 'kamis_seg', 'exec'), ns)

kamis_request = ns['kamis_request']
_kamis_err_text = ns['_kamis_err_text']
_kamis_ssl_context = ns['_kamis_ssl_context']
_kamis_session = ns['_kamis_session']

URL = "https://www.kamis.or.kr/service/price/xml.do"
PARAMS = {"action": "periodProductList"}


class FakeResp:
    def __init__(self, tag):
        self.tag = tag
        self.status_code = 200


def _install(seq):
    """seq: 라벨 -> 결과(예외 or FakeResp). 각 attempt 를 가로챈다."""
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None, verify=True):
        label = 'https' if url.startswith('https') else 'http'
        calls.append((label, verify))
        out = seq.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    class FakeSession:
        def __init__(self):
            self.adapters = {}

        def mount(self, *a, **k):
            pass

        def get(self, url, **kw):
            return fake_get(url, **kw)

        def close(self):
            pass

    ns['_requests'] = types.SimpleNamespace(get=fake_get, Session=FakeSession)
    return calls


def test_first_https_succeeds():
    _install([FakeResp('ok')])
    r = kamis_request(URL, PARAMS)
    assert r._smart_transport == 'https'


def test_legacy_https_tried_before_plain_http():
    """암호화되는 레거시 HTTPS를 평문 HTTP보다 먼저 시도해야 인증키가 덜 노출된다."""
    calls = _install([SSLError('handshake'), FakeResp('ok')])
    r = kamis_request(URL, PARAMS)
    assert r._smart_transport == 'https-legacy'
    assert [c[0] for c in calls] == ['https', 'https']


def test_http_fallback_used_when_both_https_fail():
    calls = _install([SSLError('a'), SSLError('b'), FakeResp('ok')])
    r = kamis_request(URL, PARAMS)
    assert r._smart_transport == 'http-fallback'
    assert [c[0] for c in calls] == ['https', 'https', 'http']


def test_insecure_last_resort():
    calls = _install([SSLError('a'), SSLError('b'), SSLError('c'), FakeResp('ok')])
    r = kamis_request(URL, PARAMS)
    assert r._smart_transport == 'https-insecure'
    assert calls[-1][1] is False          # 마지막 시도만 verify=False


def test_all_fail_reports_every_reason():
    _install([SSLError('e1'), SSLError('e2'), SSLError('e3'), SSLError('e4')])
    try:
        kamis_request(URL, PARAMS)
    except RuntimeError as ex:
        msg = str(ex)
    else:
        raise AssertionError('예외가 발생해야 한다')
    for lab in ('https', 'https-legacy', 'http-fallback', 'https-insecure'):
        assert lab in msg, msg
    assert 'e4' in msg


def test_err_text_pulls_out_ssl_reason():
    inner = ssl.SSLError(1, "[SSL: UNSAFE_LEGACY_RENEGOTIATION_DISABLED] unsafe legacy renegotiation disabled (_ssl.c:1000)")
    outer = SSLError("HTTPSConnectionPool(host='www.kamis.or.kr', port=443): Max retries exceeded")
    outer.__cause__ = inner
    txt = _kamis_err_text(outer)
    assert 'UNSAFE_LEGACY_RENEGOTIATION_DISABLED' in txt
    assert txt.startswith('SSLError')


def test_ssl_context_options():
    ctx = _kamis_ssl_context(verify=True, legacy=True)
    assert ctx.options & getattr(ssl, 'OP_LEGACY_SERVER_CONNECT', 0x4)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    ctx2 = _kamis_ssl_context(verify=False, legacy=True)
    assert ctx2.verify_mode == ssl.CERT_NONE
    assert ctx2.check_hostname is False


def test_real_session_mounts_adapter():
    ns['_requests'] = requests
    sess = _kamis_session(_kamis_ssl_context())
    assert 'https://' in sess.adapters and 'http://' in sess.adapters
    sess.close()
