"""Read-only, exact-v1 output fetch without traversing irrelevant output pages."""
from pathlib import Path
import time
import requests
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def fetch_required(api, ref, destination, required):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    owner, slug = ref.split('/')
    needed = set(required)
    seen = set()
    token = None
    log_saved = False
    for page in range(500):
        for attempt in range(4):
            try:
                with api.build_kaggle_client() as client:
                    request = ApiListKernelSessionOutputRequest()
                    request.user_name = owner
                    request.kernel_slug = slug
                    # API version_label is not a numeric version selector. Caller
                    # checks canonical current_version_number == 1 before/after.
                    request.page_size = 100
                    request.page_token = token
                    response = client.kernels.kernels_api_client.list_kernel_session_output(request)
                break
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        # The SDK normally postpones saving this until after recursive pagination.
        if response.log:
            (destination / (slug + '.log')).write_text(response.log, encoding='utf-8')
            log_saved = True
        for item in response.files or []:
            if item.file_name not in needed:
                continue
            target = destination / item.file_name
            target.resolve().relative_to(destination.resolve())
            seen.add(item.file_name)
            for attempt in range(4):
                try:
                    with requests.get(item.url, stream=True, timeout=(30, 180)) as download:
                        download.raise_for_status()
                        length = download.headers.get('Content-Length')
                        if target.is_file() and length and target.stat().st_size == int(length):
                            break  # Caller independently verifies the artifact hashes.
                        target.parent.mkdir(parents=True, exist_ok=True)
                        partial = target.with_name(target.name + '.partial')
                        with partial.open('wb') as stream:
                            for chunk in download.iter_content(1 << 20):
                                stream.write(chunk)
                        if length:
                            assert partial.stat().st_size == int(length), item.file_name
                        partial.replace(target)
                    break
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
        if seen == needed and log_saved:
            return {'version': 1, 'pages': page + 1, 'files': sorted(seen), 'log_saved': True}
        new_token = response.next_page_token
        assert new_token and new_token != token, ('Missing exact outputs/log', sorted(needed - seen))
        token = new_token
    raise RuntimeError('Bounded output pagination exhausted')
