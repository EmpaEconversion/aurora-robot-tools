package main

import (
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	argsWithoutProg := os.Args[1:]

	// Get the path to the directory of the current script
	dir, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		log.Fatal(err)
	}

	// Construct the path to the Python script
	pyScriptPath := filepath.Join(dir, "backup_database.py")

	// Run the Python script
	cmd := exec.Command("py", append([]string{pyScriptPath}, argsWithoutProg...)...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()

	if err != nil {
		log.Fatalf("cmd.Run() failed with %s\n", err)
	}

	os.Exit(cmd.ProcessState.ExitCode())
}
