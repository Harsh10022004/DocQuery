# Git Documentation

Source: https://git-scm.com/doc

## Everyday Branching and Switching
Git branches are lightweight movable pointers to commits.
- Create and switch to a new branch: `git checkout -b feature-branch` or `git switch -c feature-branch`.
- List local and remote branches: `git branch -a`.
- Delete a local branch: `git branch -d feature-branch` (or `-D` to force delete).

## Git Merge vs Git Rebase
Both commands integrate changes from one branch into another:
- `git merge feature`: Creates a 3-way merge commit combining histories. Preserves the exact chronological timeline of commits.
- `git rebase main`: Re-applies your feature commits on top of the tip of main. Creates a clean, linear commit history without merge commits.
*Caution*: Never rebase commits that have already been pushed to a public/shared remote repository.

## Cherry-Pick
`git cherry-pick <commit-hash>` applies the changes introduced by a specific existing commit from another branch onto your currently checked-out branch.
This is useful for backporting bug fixes to production release branches without merging entire feature branches.

## Git Stash
When you need to switch branches but have uncommitted local changes:
- Save current changes: `git stash save "WIP on auth"`
- List stored stashes: `git stash list`
- Restore latest stash: `git stash pop` (restores and removes from stash list)
- View stash contents without applying: `git stash show -p stash@{0}`

## Undoing Changes: Reset vs Revert
- `git revert <commit-hash>`: Creates a new commit that inverts the changes made by the specified commit. Safe for shared branches.
- `git reset --soft HEAD~1`: Moves HEAD back by 1 commit while leaving your modified files in the staging area.
- `git reset --hard HEAD~1`: Completely wipes out uncommitted changes and resets the working directory to the previous commit. Destructive!

## Resolving Merge Conflicts
When Git cannot automatically resolve differences between two branches:
1. Git pauses and marks conflicted files with `<<<<<<<`, `=======`, and `>>>>>>>`.
2. Open files and edit to keep desired lines.
3. Stage the resolved files: `git add <file>`.
4. Finalize the merge commit: `git commit -m "Resolve merge conflict"`.
