async function download() {

  document.querySelectorAll(".time img").forEach(dom => {
    console.log(dom.click())//遍历选中默认100张
  })
  await delay(2000) //延时2秒
  document.querySelector(".yk-icon-datuxiazai").click()//触发下载
  await delay(2000)//延迟2秒
  // document.querySelectorAll(".time img").forEach(dom=>{
  //     console.log(dom.click())//把下载的删除，我的目的是为了清空一刻相册，转到别的网盘，你不要清空这句可以删掉
  // })
  await delay(2000)
  document.querySelector(".right-btn .yk-icon-trash").click()
  await delay(2000)
  //document.querySelector(".popover-content .confirm").click()//触发删除，也可以删掉
  await delay(2000)

  setTimeout(this.download, 7000)
}
//延迟函数
function delay(time) {
  return new Promise((r, e) => {
    setTimeout(r, time)
  })
}
download()
