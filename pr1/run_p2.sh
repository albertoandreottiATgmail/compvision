#exercise 2
for i in `seq 80 5 120`;
    do
        rm  cache/*.dat
        rm  cache/*.feat
        cd cache
        rm **/*.bovw
        cd ..
        python lab1.py -c $i -t 4 -k rbf
    done  
