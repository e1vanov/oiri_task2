from server import app

import os
import shutil

if __name__ == '__main__':
    try:
        shutil.rmtree('tmp')
    except FileNotFoundError:
        pass
    os.mkdir('tmp')
    os.mkdir('tmp/images')
    app.run(host='::', port=8080)

