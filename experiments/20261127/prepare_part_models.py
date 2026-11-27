"""Download public papers/code/checkpoints into the workspace, never install them."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[2]
RESOURCES={
 "partglee_paper":("docs/papers/07_2024_PartGLEE.pdf","https://arxiv.org/pdf/2407.16696","pdf"),
 "vlpart_paper":("docs/papers/08_2023_VLPart.pdf","https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Going_Denser_with_Open-Vocabulary_Part_Segmentation_ICCV_2023_paper.pdf","pdf"),
 "partglee_code":("experiments/vendor/PartGLEE-master.zip","https://codeload.github.com/ProvenceStar/PartGLEE/zip/refs/heads/master","zip"),
 "vlpart_code":("experiments/vendor/VLPart-main.zip","https://codeload.github.com/facebookresearch/VLPart/zip/refs/heads/main","zip"),
 "clip_code":("experiments/vendor/CLIP-main.zip","https://codeload.github.com/openai/CLIP/zip/refs/heads/main","zip"),
 "vlpart_r50":("experiments/model_cache/parts/r50_pascalpart.pth","https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_pascalpart.pth","weights"),
 "partglee_r50":("experiments/model_cache/parts/PartGLEE_RN50.pth","https://drive.usercontent.google.com/download?id=1SK0HsYXBcTvprFAxqxH-lgpkgV67B4UZ&export=download&authuser=0&confirm=t","weights"),
}


def download(name):
    relative,url,kind=RESOURCES[name];path=ROOT/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    started=time.time()
    if not path.exists():
        request=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
        temp=path.with_suffix(path.suffix+".part")
        with urllib.request.urlopen(request,timeout=90) as response:
            content_type=response.headers.get("Content-Type","")
            if "text/html" in content_type and kind=="weights":
                raise RuntimeError("Checkpoint endpoint returned HTML instead of weights")
            total=0;next_progress=32*1024*1024
            with temp.open("wb") as output:
                while chunk:=response.read(4*1024*1024):
                    output.write(chunk);total+=len(chunk)
                    if total>=next_progress:
                        print(name,total,"bytes",flush=True);next_progress+=32*1024*1024
        first=temp.open("rb").read(16)
        if kind=="pdf" and not first.startswith(b"%PDF-"):
            raise RuntimeError("Not a PDF")
        if kind=="zip" and not zipfile.is_zipfile(temp):
            raise RuntimeError("Not a ZIP archive")
        if kind=="weights" and (first.lstrip().startswith(b"<") or temp.stat().st_size<1024*1024):
            raise RuntimeError("Not a plausible checkpoint")
        temp.replace(path)
    digest=hashlib.file_digest(path.open("rb"),"sha256").hexdigest()
    if kind=="zip":
        destination=ROOT/"experiments/vendor"
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target=(destination/member.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive path")
            # A completed tree is retained rather than overwritten on retries.
            tree=destination/Path(archive.infolist()[0].filename).parts[0]
            if not tree.exists():
                archive.extractall(destination)
    return dict(name=name,path=str(path.relative_to(ROOT)),source=url,status="ok",bytes=path.stat().st_size,
                sha256=digest,elapsed_seconds=round(time.time()-started,2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resources",nargs="+",choices=RESOURCES)
    args=parser.parse_args()
    log=ROOT/"experiments/20261127/part_model_resources.json"
    results=json.loads(log.read_text(encoding="utf-8")) if log.exists() else {}
    for name in args.resources:
        try:
            results[name]=download(name)
        except Exception as exc:
            results[name]=dict(name=name,status="failed",source=RESOURCES[name][1],
                               error=type(exc).__name__+": "+str(exc))
        log.write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(results[name],ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
