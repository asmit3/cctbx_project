for ((i=22; i <= 9817622; i=$i+98176))
#for ((i=22; i <= 9817622; i=$i+98000000000000000176))
do
    echo $i $1 $PWD
    nohup dials.python ../simulated_exgauss_intensities.py $i >> logfile &
done
