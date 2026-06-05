refresh_global
yes | improv cleanup
improv run naumann_simulate.yaml --tui-client-timeout 3600 --port-read-timeout 3600
setup
run
stop
quit
grep "jdg" global.log 
grep -m 10 -A 10 "error" global.log
ls


pkill -USR1 improv
pkill -9 improv
sudo /etc/init.d/redis-server stop


coverage run --save-signal=USR1 $(which improv) run naumann_simulate.yaml --tui-client-timeout 30 --port-read-timeout 30
coverage combine --keep
coverage html && firefox htmlcov/index.html 1>/dev/null 2>/dev/null

python convert_owens_data.py

mamba activate improv
export PYTHONPATH=$HOME/Documents/naumann_collab/improv/:$PYTHONPATH
export PYTHONPATH=$HOME/Documents/naumann_collab/CaImAn/:$PYTHONPATH
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# tail -f global.log | ./colorize.py

refresh_global() { cp global.old.log "/tmp/global.log.$(date +%Y%m%d_%H%M%S)"; cp global.log global.old.log; > global.log; }
