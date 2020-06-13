#! /bin/sh

###### Script writer: Shivansh Saini (@shivanshs9) [26/10/2018] ######

## Initial Configurations
REPO_URI="spectrochempy/spectrochempy"
GITHUB_USERNAME="fernandezc"
CHANGELOG_FILE="CHANGELOG.md"

## Helper functions
get_line_no() {
	release="$1"
	grep -n -m 1 "$release" "$CHANGELOG_FILE" | cut -d ":" -f 1
}

get_current_tag() {
	git describe --abbrev=0 --tags
}

get_latest_tag() {
	curl -s $API_RELEASES_URL"/latest" | python3 -c \
		"import sys, json; print(json.load(sys.stdin)['tag_name'])" 2>/dev/null
}

get_body() {
	current_local_tag="$1"
	last_deployed_tag="$2"
	start=$(get_line_no "${current_local_tag:1}")
	: $(( start++ ))
	end=$(get_line_no "${last_deployed_tag:1}")
	: $(( end-- ))

	# Getting the updated content in CHANGELOG.md and making it JSON-safe
	sed "${start},${end}p" "$CHANGELOG_FILE" -n | sed -z 's/\n/\\n/g'
}

print_help() {
	echo -e "\nUsage:"
	echo -e "\t$0 [options]"

	echo -e "\nDescription:"
	echo -e "\tA shell utility script to auto-generate release notes of configured repository based \
on the latest release tag and the current tag. Looks for changes in the configured changelog file."

	echo -e "\nOptions:"
	options="\t-h, --help|Show help.\n\
\t-u, --username <username>|Calls the Github API providing username as <username>.\n\
\t-r, --repository <repo_uri>|Configures the repository to use github <repo_uri>, which is in the format of <owner>/<repo>.\n\
\t-c, --changelog <file>|Configures to look for <file> for release notes. The release notes should start from\
the next line of the tag name."
	echo -e $options | column -t -s '|'
}

## Main script

while [[ $# -gt 0 ]]; do
	key="$1"
	case $key in
		-u|--username)
			GITHUB_USERNAME="$2"
			shift
			shift
			;;
		-r|--repository)
			REPO_URI="$2"
			shift
			shift
			;;
		-c|--changelog)
			CHANGELOG_FILE="$2"
			shift
			shift
			;;
		-h|--help)
			print_help
			exit
			;;
		*)
			echo "Invalid argument. Use $0 --help to get help."
			exit 2
			;;
	esac
done

API_RELEASES_URL="https://api.github.com/repos/"${REPO_URI}"/releases"

if [[ ! -f "$CHANGELOG_FILE" ]]; then
	echo "ERROR: \"$CHANGELOG_FILE\" file doesn't exist."
	exit 1
fi

echo -n "Getting current release tag..."
current_local_tag=$(get_current_tag)
echo " $current_local_tag"

echo -n "Getting latest release tag..."
last_deployed_tag=$(get_latest_tag)
echo " $last_deployed_tag"

if [[ -z "$last_deployed_tag" ]]; then
	echo "ERROR: Unable to find last release tag from \"$REPO_URI.\""
	exit 1
fi

if [[ "$last_deployed_tag" = "$current_local_tag" ]]; then
	echo "SKIPPING: Local tag is up-to-date with the latest release tag."
	exit
fi

body=$(get_body "$current_local_tag" "$last_deployed_tag")
title="$current_local_tag"

echo "----Releasing \"$current_local_tag\" to \"$REPO_URI\"----"

data="{\"tag_name\": \"$current_local_tag\", \"name\": \"$title\", \"body\": \"$body\"}"

curl -s -X POST -H "Content-Type:application/json" -u "$GITHUB_USERNAME" \
"$API_RELEASES_URL" --data-binary @- <<<$data
