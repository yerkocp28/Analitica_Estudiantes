// Encabezado corrido para la salida PDF (Typst).
//
// En HTML el logo va una sola vez, en una franja sobre el titulo. En un PDF
// de 13 paginas que se imprime y circula suelto, la convencion es distinta:
// el logo se repite discreto en el margen superior, de modo que cualquier
// hoja quede identificada. Por eso las dos salidas no comparten el mismo
// tratamiento del logo.
//
// Colores muestreados del propio logo, iguales a los de estilos.css:
//   rojo del escudo  #E2211C
//   carbon del texto #3D3935
//   gris de apoyo    #6B6560
//   linea            #E3E0DC

// El logo es un lockup de escudo mas dos lineas de texto en una caja de
// 330x260. Por debajo de ~12 mm de alto la bajada "UNIVERSIDAD AUTONOMA DE
// CHILE" deja de leerse y el logo se ve como una mancha, asi que la portada
// lo lleva a 18 mm y el resto de las paginas a 12 mm, que es el minimo
// legible. El techo lo pone el margen superior: el encabezado se dibuja
// DENTRO de el, y un logo mas alto que margin.top * (1 - header-ascent) se
// recorta contra el borde de la hoja.
#set page(
  header: context {
    let portada = counter(page).get().first() == 1
    grid(
      columns: (1fr, auto),
      align: (left + bottom, right + bottom),
      text(
        size: if portada { 8.5pt } else { 7.5pt },
        fill: rgb("#6B6560"),
      )[Proyecto Analítica de Estudiantes · Facultad de Administración y Negocios],
      image("assets/logo-ua.png", height: if portada { 18mm } else { 12mm }),
    )
    v(-2mm)
    line(
      length: 100%,
      stroke: (if portada { 1.6pt } else { 0.6pt })
        + (if portada { rgb("#E2211C") } else { rgb("#E3E0DC") }),
    )
  },
  header-ascent: 28%,
  footer: context {
    line(length: 100%, stroke: 0.6pt + rgb("#E3E0DC"))
    v(-1mm)
    grid(
      columns: (1fr, auto),
      align: (left + top, right + top),
      text(size: 7.5pt, fill: rgb("#6B6560"))[Universidad Autónoma de Chile],
      text(size: 7.5pt, fill: rgb("#3D3935"))[
        #counter(page).display("1 / 1", both: true)
      ],
    )
  },
  footer-descent: 25%,
)

// La numeracion de secciones toma el rojo institucional, igual que en HTML.
#show heading: it => {
  set text(fill: rgb("#3D3935"))
  it
}
