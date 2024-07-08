window.addEventListener("resize", function() {
  togglePageOverview();
});

function togglePageOverview() {
  var pageOverview = document.querySelector(".page-overview");
  var desktopMenu = document.querySelector(".desktop-menu");
  var mobileMenu = document.querySelector(".mobile-menu");
  var contentElement = document.getElementById("subheader");

  if (window.innerWidth > 1300) {
    var desktopContent = document.getElementById("desktop-content");
    desktopContent.style.display = "flex";
    pageOverview.style.display = "block";
    desktopMenu.style.display = "block";
    mobileMenu.style.display = "none";

    } else {
    var desktopContent = document.getElementById("desktop-content");
    desktopContent.style.display = "none";

    pageOverview.style.display = "none";
    desktopMenu.style.display = "none";

    mobileMenu.style.display = "block";
  }
}
togglePageOverview();
