"""Process-local SDK transport via Windows certificate-validating Schannel.

No automatic write retries, no credentials in files/argv/logs. The caller must
explicitly allow each write endpoint. This does not change scientific sources.
"""
import json,subprocess
import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient

READS={'IntrospectToken','GetKernel','GetKernelSessionStatus','ListKernelSessionOutput','ListKernels','GetDataset','GetSubmissionLimits','ListSubmissions','GetSubmission'}

def install(allowed_writes=()):
    allowed=READS|set(allowed_writes)
    def call(self,service_name,request_name,request,response_type):
        assert request_name in allowed, 'Endpoint not explicitly allowed: '+request_name
        self._init_session();prepared=self._prepare_request(service_name,request_name,request)
        assert prepared.url.startswith('https://api.kaggle.com/v1/')
        def quote(s):return '"'+str(s).replace('\\','\\\\').replace('"','\\"').replace('\r','\\r').replace('\n','\\n')+'"'
        config=['url = '+quote(prepared.url),'request = "POST"','silent','show-error','compressed','connect-timeout = 15','max-time = 120','write-out = "\\n%{http_code}"']
        for k,v in prepared.headers.items():
            if k.lower() not in ['accept-encoding','content-length']:config.append('header = '+quote(k+': '+v))
        body=prepared.body.decode() if isinstance(prepared.body,bytes) else prepared.body
        config.append('data = '+quote(body))
        result=subprocess.run(['C:/Windows/System32/curl.exe','--config','-'],input='\n'.join(config)+'\n',capture_output=True,text=True,encoding='utf-8',timeout=130)
        if result.returncode:raise RuntimeError('Schannel transport failed; do not retry writes: '+str(result.returncode)+' '+result.stderr[:300])
        body,status=result.stdout.rsplit('\n',1)
        response=requests.Response();response.status_code=int(status);response._content=body.encode('utf-8');response.headers['Content-Type']='application/json';response.url=prepared.url
        return self._prepare_response(response_type,response)
    KaggleHttpClient.call=call
