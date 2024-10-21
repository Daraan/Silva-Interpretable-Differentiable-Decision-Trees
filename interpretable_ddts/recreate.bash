#!/bin/bash
# Execute with bash not sh
source env/bin/activate
cd runfiles

# NO GPU
# MLP
export processes=0
start=`date +%s`
start_mlp=$start
for rl_env in "lunar" "cart"
do
    for i in 0 1 2 4 8 16 32
    do
        #echo "Running MLP layers: iteration $i: $rl_env"
        # Add your commands here
        python gym_runner.py -a mlp -e 1000 -env $rl_env --num_hidden $i --process_number $processes --silent --seed 0 &
        # 5x5 cores
        processes=$(($processes + 1))
        if (( (($processes % 5)) == 0 )); then
            echo "Waiting for processes to finish"
            wait
        fi
    done
done
# end time does not account still running processes
end_mlp=`date +%s`
start_ddt=$end_mlp

# DDT
for rl_env in "lunar" "cart"
do
    for i in 2 4 8 16 32
    do
        #echo "Running DDT iteration $i: $rl_env"
        # Add your commands here
        start=`date +%s`
        python gym_runner.py -a ddt -e 1000 -env $rl_env --num_leaves $i --process_number $processes --silent --seed 0 &
        end=`date +%s`
        # 5x5 cores
        processes=$(($processes + 1))
        if (( (($processes % 5)) == 0 )); then
            echo "Waiting for processes to finish"
            wait
        fi
    done
done
# end time does not account still running processes
end_ddt=`date +%s`
start_rule_list=$end_ddt

# DDT rule list
for rl_env in "lunar" "cart"
do
    for i in 2 4 8 16 32
    do
        #echo "Running DDT Rule list iteration $i: $rl_env"
        # Add your commands here
        start=`date +%s`
        python gym_runner.py -a ddt -e 1000 -env $rl_env --num_leaves $i  --rule_list --process_number $processes --silent --seed 0 &
        #echo "proceeses $processes"
        end=`date +%s`
        # 5x5 cores
        #processes=$(($processes + 1))
        if (( (($processes % 5)) == 0 )); then
            echo "Waiting for processes to finish"
            wait
        fi
    done
done
wait
end_rule_list=`date +%s`
end=$end_rule_list

# Runtimes
echo "\---------------------------------/"
echo "MLP: $((end_mlp - start_mlp))s"
echo "DDT: $((end_ddt - start_ddt))s"
echo "Rule List: $((end_rule_list - start_rule_list))s"
echo "Total: $((end - start))s"

# Discrete
# This will take much longer
start=`date +%s`
python run_discrete_agent.py -env cart -f -r -d --all
end=`date +%s`
echo "Discrete Testing cart took: $((end - start))s"

start=`date +%s`
python run_discrete_agent.py -env lunar -f -r -d --all
end=`date +%s`
echo "Discrete Testing lunar took: $((end - start))s"
