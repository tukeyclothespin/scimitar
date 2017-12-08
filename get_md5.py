import hashlib
from base64 import b64encode

def file_content_hash(in_filename):
    # Get MD5 hash of file contents
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(in_filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)

    return b64encode(hasher.digest()).decode('ascii')