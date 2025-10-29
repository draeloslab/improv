improv cleanup
coverage run --save-signal=USR1 $(which improv) run savier_simulate.yaml
stop
quit
pkill -USR1 improv
improv cleanup
# pkill -9 improv

coverage combine
coverage html && open htmlcov/index.html 1>/dev/null 2>/dev/null


python -m ipdb $(which improv) run savier_simulate.yaml
python -m ipdb $(which improv) server -c 0 -o 0 -l 0 -f global.log savier_simulate.yaml
