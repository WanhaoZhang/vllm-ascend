"""Bounded exact cache for causal diagnosis, never an accepted implementation.

Keeps official packed weights resident and caches only eight layers per rank.
FP4 E2M1 nibbles become the exact E4M3 bit patterns; E8M0 stays unchanged.
"""
import json
import os
import numpy as np
import torch

_CACHE = {}
_LIMIT = int(os.environ.get('DIAGNOSTIC_CACHE_LAYERS', '8'))
_LUT8 = np.asarray([0x00,0x30,0x38,0x3c,0x40,0x44,0x48,0x4c,
                    0x80,0xb0,0xb8,0xbc,0xc0,0xc4,0xc8,0xcc], dtype=np.uint8)
_LUT16 = np.asarray([int(_LUT8[b&15]) | (int(_LUT8[b>>4])<<8)
                     for b in range(256)], dtype='<u2')

def exact_cache(w1, s1, w2, s2):
    if w1.dtype != torch.uint8 or _LIMIT == 0:
        return w1,s1,w2,s2
    key = w1.data_ptr()
    if key in _CACHE:
        return _CACHE[key][1]
    if len(_CACHE) >= _LIMIT:
        return w1,s1,w2,s2
    def unpack(weight):
        raw = weight.transpose(1,2)
        E,N,halfK = raw.shape
        output_bytes = torch.empty((E,N,halfK*2), dtype=torch.uint8, device=weight.device)
        for e in range(E):
            cpu = raw[e].cpu().numpy()
            expanded = _LUT16[cpu].view(np.uint8).reshape(N,halfK*2)
            output_bytes[e].copy_(torch.from_numpy(expanded))
        return output_bytes.view(torch.float8_e4m3fn)
    v1, v2 = unpack(w1), unpack(w2)
    ns1 = s1.transpose(1,2).reshape(v1.shape[0],v1.shape[1],v1.shape[2]//32)
    ns2 = s2.transpose(1,2).reshape(v2.shape[0],v2.shape[1],v2.shape[2]//32)
    result = (v1,ns1,v2,ns2)
    _CACHE[key] = ((w1,s1,w2,s2), result)
    # Independent CPU floating-point LUT confirms every nibble mapping.
    expected = torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    assert np.array_equal(expected,_LUT8)
    print('DIAGNOSTIC_EXACT_CACHE '+json.dumps(dict(layer=len(_CACHE), rank=int(os.environ.get('RANK',-1)),
          device=str(w1.device), additional_bytes=v1.numel()+v2.numel(), exact_mapping=True)),flush=True)
    return result
