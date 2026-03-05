import runpy, traceback
try:
    runpy.run_path('00_pytorch_fundamentals.py', run_name='__not_main__')
    print('MODULE OK')
except Exception:
    traceback.print_exc()
