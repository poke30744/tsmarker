import shutil, subprocess, tempfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import tscutter.ffmpeg
from tscutter.common import InvalidTsFormat

class InputFile(tscutter.ffmpeg.InputFile):
    def ExtractArea(self, area, folder, ss, to, fps='1/1', quiet=False):
        folder = self.path.with_suffix('') if folder is None else Path(folder)
        if folder.is_dir():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

        info = self.GetInfo()
        w, h, x, y = int(round(area[2] * info['width'])), int(round(area[3] * info['height'])), int(round(area[0] * info['width'])), int(round(area[1] * info['height']))
        args = [ 'ffmpeg', '-hide_banner' ]
        if ss is not None and to is not None:
            args += [ '-ss', str(ss), '-to', str(to) ]
        fpsStr = ',fps={}'.format(fps) if fps else ''
        args += [
            '-i', self.path,
            '-filter:v', 'crop={}:{}:{}:{}{}'.format(w, h, x, y, fpsStr),
            '{}/out%8d.bmp'.format(folder) ]
        pipeObj = subprocess.Popen(args, stderr=subprocess.PIPE, universal_newlines='\r', errors='ignore')
        if to > info['duration']:
            to = info['duration']
        with tqdm(total=to - ss, disable=quiet, unit='secs') as pbar:
            pbar.set_description('Extracting area')
            for line in pipeObj.stderr:
                if 'time=' in line:
                    for item in line.split(' '):
                        if item.startswith('time='):
                            timeFields = item.replace('time=', '').split(':')
                            time = float(timeFields[0]) * 3600 + float(timeFields[1]) * 60  + float(timeFields[2])
                            pbar.update(time - pbar.n)
            pbar.update(to - ss - pbar.n)
        pipeObj.wait()
    
    
    def ExtractLogo(self, area, outputPath, ss=0, to=999999, fps='1/1', quiet=False):
        outputPath = self.path.parent / (self.path.stem + '_logo.png') if outputPath is None else Path(outputPath)
        with tempfile.TemporaryDirectory(prefix='logo_pics_') as tmpLogoFolder:
            self.ExtractArea(area=area, folder=tmpLogoFolder, ss=ss, to=to, fps=fps, quiet=quiet)
            pics = list(Path(tmpLogoFolder).glob('*.bmp'))
            picSum = None
            for path in tqdm(pics, desc='Loading pics', total=len(pics), disable=quiet):
                image = np.array(Image.open(path)).astype(np.float32)
                picSum = image if picSum is None else (picSum + image)
            if picSum is None:
                raise InvalidTsFormat(f'"{self.path.name}" is invalid!')
        picSum /= len(pics)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(picSum.astype(np.uint8)).save(str(outputPath))
        return outputPath