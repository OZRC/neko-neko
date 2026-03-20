#bin/bash
function choose_from_menu() {
	local prompt="$1" outvar="$2"
	shift
	shift
	local options=("$@")
	options+=("cancel")
	local cur=0 count=${#options[@]} index=0
	local esc=$(echo -en "\e") # cache ESC as test doesn't allow esc codes
	printf "$prompt\n_______________\n"
	while true; do
		# list all options (option list is zero-based)
		index=0
		for o in "${options[@]}"; do
			if [[ "$index" == "$cur" ]]; then
				echo -e " > \e[36m$o\e[0m" # mark & highlight the current option
			else
				echo -e "   \e[2m$o\e[0m"
			fi
			((index++))
		done
		read -s -n3 key               # wait for user to key in arrows or ENTER
		if [[ $key == $esc[A ]]; then # up arrow
			((cur--))
			((cur < 0)) && ((cur = 0))
		elif [[ $key == $esc[B ]]; then # down arrow
			((cur++))
			((cur >= count)) && ((cur = count - 1))
		elif [[ $key == "" ]]; then # nothing, i.e the read delimiter - ENTER
			break
		fi
		echo -en "\e[${count}A" # go up to the beginning to re-render
	done

	if [[ "${options[$cur]}" = "cancel" ]]; then
		echo -e "\e[31mExitting...\e[0m"
		exit 0
	fi

	# export the selection to the requested output variable
	printf -v $outvar "${options[$cur]}"
}

action=("pycache" "logs")
choose_from_menu "Select a method:" selected_action "${action[@]}"

if [[ $selected_action = "pycache" ]]; then
    find . | grep -E "(.~lock.*.xlsx#$|__pycache__|\.pyc|\.pyo$)" | xargs rm -rf;
    echo -e "\e[32mAll __pycache__ and .pyc files have been removed.\e[0m"
elif [[ $selected_action = "log" ]]; then
    find . | grep -E "(\.log|\.logs$)" | xargs rm -rf;
    echo -e "\e[32mAll .log and .logs files have been removed.\e[0m"
fi
