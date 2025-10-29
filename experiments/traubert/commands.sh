echo -n '' > global.log
improv cleanup
improv run traubert_simulate.yaml # coverage run --save-signal=USR1 $(which improv) run traubert_simulate.yaml
stop
quit
# pkill -USR1 improv
improv cleanup
pkill -9 improv

coverage combine
coverage html && open htmlcov/index.html 1>/dev/null 2>/dev/null


python -m ipdb $(which improv) run traubert_simulate.yaml
python -m ipdb $(which improv) server -c 0 -o 0 -l 0 -f global.log traubert_simulate.yaml

improv run traubert_simulate.yaml
