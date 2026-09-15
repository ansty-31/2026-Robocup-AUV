"""Run final checks and package only deliverable files (no caches or scratch)."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import subprocess
import sys
import zipfile

workspace=Path(__file__).resolve().parents[1]
root=workspace/'gate_pnp'
previous=workspace/'work/gate_pnp-before-guidance.zip'
archive=workspace/'gate_pnp.zip'
hash_bytes=lambda b:hashlib.sha256(b).hexdigest()
sources=['config/front_camera.yaml','config/vision.yaml','reference/prepare_frames(1).py']
source_checks={}
with zipfile.ZipFile(previous) as old:
    for name in sources:
        original=old.read('gate_pnp/'+name)
        current=(root/name).read_bytes()
        assert original==current, f'Source attachment changed: {name}'
        source_checks[name]={'sha256':hash_bytes(current),'identical_to_previous_delivery':True}

command=[sys.executable,'-B','-m','unittest','discover','-s','tests','-v']
run=subprocess.run(command,cwd=root,capture_output=True,text=True,encoding='utf-8')
log=run.stdout+run.stderr
(root/'test-results.txt').write_text(log,encoding='utf-8')
if run.returncode:
    print(log)
    raise SystemExit(run.returncode)
count=int(re.search(r'Ran (\d+) tests',log).group(1))
entries={}
for mode,args in [('pnp',[]),('guidance',['--guidance'])]:
    result=subprocess.run([sys.executable,'-B','-m','gate_pnp',*args],cwd=root,
                          capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr+result.stdout
    entries[mode]=json.loads(result.stdout)
assert entries['pnp']['status']=='NO_INPUT' and entries['pnp']['pose'] is None
assert entries['guidance']['phase']=='ACQUIRE' and entries['guidance']['action']=='HOLD'
assert entries['guidance']['pnp_result'] is None
previous_verification=root/'verification.json'
backup=workspace/'work/verification-before-guidance.json'
if not backup.exists():
    backup.write_bytes(previous_verification.read_bytes())
verification={
    'status':'IMPLEMENTED_AND_AUTOMATED_TESTS_PASSED',
    'verified_at_utc':datetime.now(timezone.utc).isoformat(),
    'python':sys.version,'interpreter':sys.executable,
    'packages':{p:importlib.metadata.version(p) for p in ('numpy','opencv-python','PyYAML')},
    'ultralytics_installed':importlib.util.find_spec('ultralytics') is not None,
    'tests':{'count':count,'exit_code':run.returncode,'command':command,
             'working_directory':str(root),'log':'test-results.txt'},
    'no_input_entrypoints':entries,'attachment_preservation':source_checks,
    'independent_review':{'scope':'Vertical recovery enum, decision branches and behavior tests',
        'findings':0,'assessment':'passed','behavior_tests_passed':25,
        'reviewed_files':['gate_pnp/guidance_types.py','gate_pnp/guidance.py','tests/test_guidance.py']},
    'previous_delivery_review':{'scope':'Initial guidance module before vertical recovery',
        'findings':2,'fixed':2,'scoped_recheck':'passed',
        'fixes':['Malformed non-target detections produce ERROR, not a no-gate scan condition',
                 'Invalid target_index produces HOLD and cannot leak nonfinite JSON metadata']},
    'verified_behaviors':['Original 32 PnP/adapter regression tests',
        'Unicode-path calibration reading without source modification',
        'Partial-gate extraction and explicit multi-target selection',
        'Recovery debounce, frame freshness/order, alternating search',
        'Only semantic TL/TR visible descends; only BR/BL visible ascends',
        'Vertical recovery requires both 0.2 seconds and three distinct valid frames',
        'Top/bottom condition changes and input faults clear vertical confirmation',
        'All 16 visibility combinations, confidence boundaries and invalid coordinates',
        'Semantic keypoint identity is not reassigned from pixel height',
        'Too-close BACKWARD takes priority over both vertical recovery actions',
        'Alignment loss permits vertical recovery after fresh confirmation',
        'PASSING excludes both vertical actions, including recovery after input faults',
        'Stable PnP then alignment; alignment loss re-enters recovery',
        'Stable centering and direction before a single START_PASS',
        'PASSING never recovers view on valid large/partial/no-gate observations',
        'Input errors HOLD without leaving PASSING; lifecycle resets explicit',
        'Result-double extraction through real OpenCV PnP to guidance decisions'],
    'not_verified':['Actual YOLO model inference','Physical actuator commands',
        'Camera-to-body extrinsic calibration','Hardware motion and actual gate crossing',
        'Underwater position/attitude accuracy','Real-data tuning of guidance thresholds',
        'OpenCV version actually used for existing training images'],
}
previous_verification.write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def deliverable_files():
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and not {'__pycache__','work'}.intersection(p.relative_to(root).parts)
                  and p.suffix!='.pyc')
files=deliverable_files()
manifest={p.relative_to(root).as_posix():hash_bytes(p.read_bytes())
          for p in files if p.name!='sha256sums.json'}
(root/'sha256sums.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in deliverable_files():
        z.write(path,'gate_pnp/'+path.relative_to(root).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,digest in manifest.items():
        assert hash_bytes(z.read('gate_pnp/'+name))==digest
    file_count=len(z.namelist())
print(json.dumps({'tests_passed':count,'original_attachments_unchanged':True,
                  'pnp_entrypoint':entries['pnp']['status'],'guidance_entrypoint':entries['guidance']['action'],
                  'archive':str(archive),'archive_files':file_count,'archive_integrity':'passed',
                  'archive_sha256':hash_bytes(archive.read_bytes())},ensure_ascii=True,indent=2))
