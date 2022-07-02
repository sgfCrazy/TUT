# GIT 学习

## 远程仓库

* 使用github作为自己的远程仓库时使用ssh进行连接，因此需要首先创建一个github账号，然后在本地生成ssh公钥，并放到github账号中。

  ```
  # 首先在本地生成一个公钥，点击右键git bash，在弹出的cmd窗口中输入下面的命令，然后一路回车
  ssh-keygen -t rsa -C "youremail@example.com"
  # windows系统生成的公钥默认放在C:\Users\${用户名}\.ssh目录下，拷贝该目录下的id_rsa.pub中的内容。之后在github上点击settings->SSH and GPG keys->new SSH keys，然后将拷贝的公钥放入输入框中。
  ```

* 在github上创建一个仓库

* 将其与本地仓库关联起来，并推送本地分支到远程仓库

  ```
  # origin是对远程仓库的别名，可以改为其他如 github_TUT等
  git remote add origin https://github.com/sgfCrazy/TUT.git
  # 将本地master分支推送到origin对应的远程仓库中的master分支。加了-u参数，会将本地master分支和远程仓库关联起来，下次可以直接git push
  git push -u origin master  
  ```

* 查看远程仓库

  ```
  git remote -v
  ```

* 删除远程仓库

  ```
  git remote rm origin
  ```

* 从远程仓库克隆

  ```
  # github 支持多种协议，包括http和ssh，ssh相对速度较快
  git clone https://github.com/sgfCrazy/TUT.git
  ```

  

