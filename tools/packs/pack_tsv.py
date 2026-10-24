# name dtype shape (one line per tensor, sorted) from the safetensors files of a pack dir.
import json, struct, sys, glob, os
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.safetensors'))):
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == '__metadata__':
            continue
        rows.append('%s\t%s\t%s' % (k, v['dtype'], ','.join(map(str, v['shape']))))
rows.sort()
sys.stdout.write('\n'.join(rows) + '\n')
