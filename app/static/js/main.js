let menu = document.querySelector('#menu-bar');
let nav = document.querySelector('.nav');

menu.onclick = () => {
  menu.classList.toggle('fa-times');
  nav.classList.toggle('active');
}

let section = document.querySelectorAll('section');
let navLinks = document.querySelectorAll('header .nav a');

window.onscroll = () => {

  menu.classList.remove('fa-times');
  nav.classList.remove('active');

  section.forEach(sec => {

    let top = window.scrollY;
    let height = sec.offsetHeight;
    let offset = sec.offsetTop - 150;
    let id = sec.getAttribute('id');

    if (top >= offset && top < offset + height) {
      navLinks.forEach(links => {
        links.classList.remove('active');
        document.querySelector('header .nav a[href*=' + id + ']').classList.add('active');
      });
    };
  });

}

const realFileBtn = document.getElementById("real-file");
const customBtn = document.getElementById("custom-button");
const fileUploadIndicator = document.getElementById("file-upload-indicator");
const fileNameDisplay = document.getElementById("file-name-display");

customBtn.addEventListener("click", function () {
  realFileBtn.click();
});

realFileBtn.addEventListener("change", function () {
  if (realFileBtn.value) {
    // Extract filename from the path
    const fileName = realFileBtn.value.match(/[\/\\]([\w\d\s\.\-\(\)]+)$/);
    const displayName = fileName ? fileName[1] : realFileBtn.value;

    // Update the file name display
    fileNameDisplay.innerHTML = '<i class="fas fa-file-image" style="margin-right: 8px;"></i>' + displayName;

    // Show the file upload indicator with animation
    fileUploadIndicator.style.display = "block";

    // Update the button text to show file is selected
    customBtn.innerHTML = '<i class="fas fa-check"></i> Change File';
    customBtn.style.background = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)";
  } else {
    // Hide indicator if no file selected
    fileUploadIndicator.style.display = "none";
    customBtn.innerHTML = "Choose A File";
    customBtn.style.background = "";
  }
});
